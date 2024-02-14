"""
Some of the helper functions are taken from the original DiT repository.
https://github.com/facebookresearch/DiT
"""
import argparse
import numpy as np
import os
import os.path as osp
import torch
import tqdm
import torch.nn.functional as F
from PIL import Image
from diffusers.models import AutoencoderKL
from diffusion.datasets import get_target_dataset
from diffusion.utils import LOG_DIR
from DiT.diffusion import create_diffusion
from DiT.diffusion.diffusion_utils import discretized_gaussian_log_likelihood
from DiT.diffusion.gaussian_diffusion import mean_flat
from DiT.download import find_model
from DiT.models import DiT_XL_2
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(image_size):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def eval_prob(unet, latent, cond_emb, diffusion, args, all_noise=None):
    assert all_noise is not None
    cond_embeddings = cond_emb.repeat(args.batch_size)
    n_train_timesteps = diffusion.num_timesteps
    assert n_train_timesteps % args.t_interval == 0

    dim = int(args.image_size) // 8
    if args.spatial:
        pred_errors = torch.zeros(n_train_timesteps // args.t_interval * args.n_trials, dim, dim, device='cpu')
    elif args.save_eps:
        pred_errors = torch.zeros(n_train_timesteps // args.t_interval * args.n_trials, 4, dim, dim, device='cpu')
    elif args.save_eps_and_var:
        pred_errors = torch.zeros(n_train_timesteps // args.t_interval * args.n_trials, 8, dim, dim, device='cpu')
    elif args.save_vb:
        pred_errors = torch.zeros(n_train_timesteps // args.t_interval * args.n_trials, 4, device='cpu')
    else:
        pred_errors = torch.zeros(n_train_timesteps // args.t_interval * args.n_trials, device='cpu')

    if all_noise is None:
        all_noise = torch.randn((args.n_trials, 4, dim, dim), device=latent.device)
    if args.dtype == 'float16':
        all_noise = all_noise.half()

    ts = []
    for t in range(args.t_interval // 2, n_train_timesteps, args.t_interval):
        ts.extend([t] * args.n_trials)
    with torch.inference_mode():
        idx = 0
        for _ in tqdm.trange(len(ts) // args.batch_size + int(len(ts) % args.batch_size != 0), leave=False):
            batch_ts = torch.tensor(ts[idx: idx + args.batch_size]).to(latent.device)
            if args.randomize_noise:
                noise = all_noise[idx: idx + len(batch_ts)]
            else:
                noise = all_noise[:len(batch_ts)]  # note: implementation has changed from the original
            latent_ = latent.repeat(len(batch_ts), 1, 1, 1)
            noised_latent = diffusion.q_sample(latent_, batch_ts, noise)

            t_input = batch_ts.to(device).half() if args.dtype == 'float16' else batch_ts.to(device)
            cond_input = cond_embeddings if len(cond_embeddings) == len(batch_ts) else cond_embeddings[:len(batch_ts)]
            model_output = unet(noised_latent, t_input, y=cond_input)
            B, C = noised_latent.shape[:2]
            noise_pred, model_var_values = torch.split(model_output, C, dim=1)

            if args.spatial:
                error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=1)
            elif args.save_eps:
                error = noise_pred
            elif args.save_eps_and_var:
                error = model_output
            elif args.save_vb:
                # compute MSE
                mse = mean_flat((noise - noise_pred) ** 2)
                l1_loss = mean_flat(torch.abs(noise - noise_pred))

                # VB
                true_mean, _, true_log_variance_clipped = diffusion.q_posterior_mean_variance(
                    x_start=latent_, x_t=noised_latent, t=t_input
                )
                dummy_model = lambda *args, r=model_output: r
                out = diffusion.p_mean_variance(
                    dummy_model, noised_latent, t_input, clip_denoised=False, model_kwargs=None
                )
                kl = normal_kl(
                    true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
                )
                kl = mean_flat(kl) / np.log(2.0)

                # NLL
                decoder_nll = -discretized_gaussian_log_likelihood(
                    latent_, means=out["mean"], log_scales=0.5 * out["log_variance"]
                )
                decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

                error = torch.cat([mse.unsqueeze(1),
                                   l1_loss.unsqueeze(1),
                                   kl.unsqueeze(1),
                                   decoder_nll.unsqueeze(1)], dim=1)
            else:
                error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)
    return pred_errors.view(n_train_timesteps // args.t_interval, args.n_trials, *pred_errors.shape[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', type=int, default=0, help='Class to use')
    parser.add_argument('--t_interval', type=int, default=5, help='Timestep interval')

    # dataset args
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'objectnet', 'imagenetv2', 'imagenetA'],
                        help='Dataset to use')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test', 'trainval'], help='Name of split')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    # run args
    parser.add_argument('--spatial', action='store_true', help='save 64x64 error maps')
    parser.add_argument('--save_eps', action='store_true', help='save 4x64x64 epsilon predictions')
    parser.add_argument('--save_eps_and_var', action='store_true', help='save 4x64x64 epsilon predictions and variance')
    parser.add_argument('--save_vb', action='store_true', help='save MSE, KL, and NLL')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--imgs_to_eval', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0, help='Random seed')  # todo: actually use this
    parser.add_argument('--noise_path', type=str, required=True, help='Path to shared noise to use')
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')
    parser.add_argument('--dtype', type=str, default='float32', help='Model data type to use')
    parser.add_argument('--randomize_noise', action='store_true', help='If True, use different noise for each t')
    parser.add_argument('--extra', type=str, default=None, help='to add to the dataset name')

    args = parser.parse_args()
    assert sum([args.spatial, args.save_eps, args.save_eps_and_var, args.save_vb]) <= 1, "Can only save one at a time"

    dataset = args.dataset
    name = f"DiT{args.image_size}x{args.image_size}_cls{args.cls}_t{args.t_interval}_{args.n_trials}trials"
    if args.randomize_noise:
        name += '_randnoise'
    extra = args.extra if args.extra is not None else ''
    if len(extra) > 0:
        run_folder = osp.join(LOG_DIR, dataset + '_' + extra, name)
    else:
        run_folder = osp.join(LOG_DIR, dataset, name)
    if args.spatial:
        run_folder += '_spatial'
    elif args.save_eps:
        run_folder += '_eps'
    elif args.save_eps_and_var:
        run_folder += '_epsvar'
    run_folder = osp.join(run_folder, args.split) if 'imagenet' in args.dataset else run_folder
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')

    # set up dataset
    dataset = get_target_dataset(args.dataset,
                                 train=args.split == 'train',
                                 transform=get_transform(args.image_size))
    image_idxs = list(range(len(dataset)))

    if args.subset_path is not None:
        subset = np.load(args.subset_path)
        image_idxs = list(subset)
    if args.imgs_to_eval is not None:
        image_idxs = image_idxs[:args.imgs_to_eval]

    vae_model = "stabilityai/sd-vae-ft-ema"
    image_size = args.image_size
    latent_size = int(image_size) // 8
    # Load model:
    model = DiT_XL_2(input_size=latent_size).to(device)
    state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state_dict)
    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)

    seed = 0
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    # TODO: may get benefits from torch.compile or fp16, but not implemented

    # load noise
    if args.noise_path is not None:
        all_noise = torch.load(args.noise_path).to(device)
        print('Loaded noise from', args.noise_path)
    else:
        raise RuntimeError('Need to specify consistent noise path, otherwise accuracy will really suffer')

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    assert 0 <= args.cls < 1000
    cls_tensor = torch.tensor([args.cls]).to(device)
    for img_idx in tqdm.tqdm(image_idxs):
        fname = osp.join(run_folder, f'{img_idx}.pt')
        # skip if we've already computed this example for this class
        if os.path.exists(fname):
            print('Skipping', img_idx)
            continue

        image = dataset[img_idx][0]
        with torch.no_grad():
            img_input = image.to(device).unsqueeze(0)
            if args.dtype == 'float16':
                assert (False, "Not sure if DiT supports half precision")
                img_input = img_input.half()
            x0 = vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215
        pred_errors = eval_prob(model, x0, cls_tensor, diffusion, args, all_noise)
        torch.save(pred_errors, fname)


if __name__ == '__main__':
    main()
