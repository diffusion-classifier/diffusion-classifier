import argparse
import clip
import numpy as np
import os
import os.path as osp
import open_clip
import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusion.utils import LOG_DIR, TOKEN_PATH
from diffusion.models import get_sd_model, get_scheduler_config
from eval_prob_adaptive import get_transform, INTERPOLATIONS
from tqdm import tqdm, trange
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"


def eval_prob(unet, latent, cond_emb, scheduler, args, all_noise=None):
    # disallow since the comparison between different
    # (image, caption) pairs would be high variance
    assert all_noise is not None
    text_embeddings = cond_emb.repeat(args.batch_size, 1, 1)
    scheduler_config = get_scheduler_config(args)
    n_train_timesteps = scheduler_config['num_train_timesteps']
    assert n_train_timesteps % args.t_interval == 0

    if args.spatial:
        pred_errors = torch.zeros(n_train_timesteps // args.t_interval * args.n_trials, 64, 64, device='cpu')
    elif args.save_eps:
        pred_errors = torch.zeros(n_train_timesteps // args.t_interval * args.n_trials, 4, 64, 64, device='cpu')
    else:
        pred_errors = torch.zeros(n_train_timesteps // args.t_interval * args.n_trials, device='cpu')

    ts = []
    for t in range(args.t_interval // 2, n_train_timesteps, args.t_interval):
        ts.extend([t] * args.n_trials)

    if all_noise is None:
        size = len(ts) if args.randomize_noise else args.batch_size
        all_noise = torch.randn((size, 4, 64, 64), device=latent.device)
    if args.dtype == 'float16':
        all_noise = all_noise.half()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

    with torch.inference_mode():
        idx = 0
        for _ in trange(len(ts) // args.batch_size + int(len(ts) % args.batch_size != 0), leave=False):
            batch_ts = torch.tensor(ts[idx: idx + args.batch_size])
            if args.randomize_noise:
                noise = all_noise[idx: idx + len(batch_ts)]
            else:
                noise = all_noise[:len(batch_ts)]  # note: implementation has changed from the original
            noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                            noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
            t_input = batch_ts.to(device).half() if args.dtype == 'float16' else batch_ts.to(device)
            text_input = text_embeddings if len(text_embeddings) == len(batch_ts) else text_embeddings[:len(batch_ts)]
            noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
            if args.guidance_scale != 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
            if args.spatial:
                error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=1)
            elif args.save_eps:
                error = noise_pred
            else:
                error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)

    # weighting = scheduler.betas ** 2 / (2 * scheduler.alphas * (1 - scheduler.alphas_cumprod))
    return pred_errors.view(n_train_timesteps // args.t_interval, args.n_trials, *pred_errors.shape[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sd', choices=('sd', 'clip', 'openclip'),
                        help='whether to use CLIP or Stable Diffusion')
    parser.add_argument('--version', type=str, default='2-0',
                        choices=clip.available_models() + ['2-1', '2-0', '1-4', 'ViT-H-14'],
                        help='model version (for sd/clip/etc)')
    parser.add_argument('--t_interval', type=int, default=5, help='Timestep interval')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--dtype', type=str, default='float16', help='Model data type to use')
    parser.add_argument('--randomize_noise', action='store_true', help='If True, use different noise for each t')
    parser.add_argument('--interpolation', type=str, default='bilinear', help='resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='to add to the dataset name')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    args = parser.parse_args()
    args.zero_noise = args.spatial = args.save_eps = False
    args.guidance_scale = 1
    preprocess = get_transform(INTERPOLATIONS[args.interpolation])

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # make folder
    if args.model in {'clip', 'openclip'}:
        name = args.model + '_' + args.version.replace('/', '_')
    elif args.model == 'sd':
        name = f'sd_v{args.version}_t{args.t_interval}_{args.n_trials}trials'
    else:
        raise NotImplementedError
    if args.randomize_noise:
        name += '_randnoise'
    if args.extra is not None:
        run_folder = osp.join(LOG_DIR, 'winoground_' + args.extra, name)
    else:
        run_folder = osp.join(LOG_DIR, 'winoground', name)
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')

    # load model
    if args.model == 'clip':
        model, preprocess = clip.load(args.version, device=device)
    elif args.model == 'openclip':
        model, _, preprocess = open_clip.create_model_and_transforms(args.version,
                                                                     pretrained='laion2b_s32b_b79k',
                                                                     device=device)
        tokenizer = open_clip.get_tokenizer(args.version)
    elif args.model == 'sd':
        # load pretrained models
        vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
        vae = vae.to(device)
        text_encoder = text_encoder.to(device)
        unet = unet.to(device)
    else:
        raise NotImplementedError
    torch.backends.cudnn.benchmark = True

    # load noise
    if args.noise_path is not None:
        assert not args.zero_noise
        all_noise = torch.load(args.noise_path).to(device)
        print('Loaded noise from', args.noise_path)
    else:
        all_noise = None

    # set up dataset
    with open(TOKEN_PATH, 'r') as f:
        token = f.read().strip()
    dataset = load_dataset("facebook/winoground", use_auth_token=token)["test"]

    # run eval
    all_scores = []
    all_tags = []
    for example in tqdm(dataset):
        # check if we've already evaluated this example
        fname = osp.join(run_folder, f'{example["id"]}.pt')
        all_tags.append(example['collapsed_tag'])
        if osp.exists(fname):
            print(f'Already evaluated {example["id"]}')
            all_scores.append(torch.load(fname))
            continue
        # evaluate every pair
        if args.model == 'sd':
            image0 = preprocess(example["image_0"])
            image1 = preprocess(example["image_1"])
            text0 = tokenizer([example["caption_0"]], padding="max_length", max_length=tokenizer.model_max_length,
                              truncation=True, return_tensors="pt")
            text1 = tokenizer([example["caption_1"]], padding="max_length", max_length=tokenizer.model_max_length,
                              truncation=True, return_tensors="pt")

            with torch.no_grad():
                img_input0 = image0.to(device).unsqueeze(0)
                img_input1 = image1.to(device).unsqueeze(0)
                if args.dtype == 'float16':
                    img_input0 = img_input0.half()
                    img_input1 = img_input1.half()
                x0 = vae.encode(img_input0).latent_dist.mean
                x1 = vae.encode(img_input1).latent_dist.mean
                x0 *= 0.18215
                x1 *= 0.18215
                text_emb0 = text_encoder(text0.input_ids.to(device))[0]
                text_emb1 = text_encoder(text1.input_ids.to(device))[0]
            results = {
                "c0_i0": eval_prob(unet, x0, text_emb0, scheduler, args, all_noise),
                "c0_i1": eval_prob(unet, x1, text_emb0, scheduler, args, all_noise),
                "c1_i0": eval_prob(unet, x0, text_emb1, scheduler, args, all_noise),
                "c1_i1": eval_prob(unet, x1, text_emb1, scheduler, args, all_noise)
            }
        elif args.model == 'clip':
            images = torch.cat([preprocess(example["image_0"].convert("RGB")).unsqueeze(0),
                                preprocess(example["image_1"].convert("RGB")).unsqueeze(0)], dim=0).to(device)
            captions = clip.tokenize([example["caption_0"], example["caption_1"]]).to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = model(images, captions)
                # This improves performance. I think it's because it allows implicitly to use text score
                # to improve image score by decreasing the wrong image-caption score. This is against the
                # rules of Winoground since the score should only depend on the image-caption pair.
                # logits_per_image = logits_per_image.softmax(dim=-1).cpu().numpy()
            results = {
                "c0_i0": logits_per_image[0, 0],
                "c0_i1": logits_per_image[1, 0],
                "c1_i0": logits_per_image[0, 1],
                "c1_i1": logits_per_image[1, 1]
            }
        elif args.model == 'openclip':
            images = torch.cat([preprocess(example["image_0"].convert("RGB")).unsqueeze(0),
                                preprocess(example["image_1"].convert("RGB")).unsqueeze(0)], dim=0).to(device)
            captions = tokenizer([example["caption_0"], example["caption_1"]]).to(device)
            with torch.no_grad():
                image_features, text_features, logit_scale = model(images, captions)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                logits_per_image = logit_scale * image_features @ text_features.t()
            results = {
                "c0_i0": logits_per_image[0, 0],
                "c0_i1": logits_per_image[1, 0],
                "c1_i0": logits_per_image[0, 1],
                "c1_i1": logits_per_image[1, 1]
            }
        else:
            raise NotImplementedError
        # save results
        torch.save(results, fname)
        all_scores.append(results)

    # adjust
    if args.model == 'sd':
        for result in all_scores:
            result["c0_i0"] = - result["c0_i0"].mean().item()
            result["c0_i1"] = - result["c0_i1"].mean().item()
            result["c1_i0"] = - result["c1_i0"].mean().item()
            result["c1_i1"] = - result["c1_i1"].mean().item()

    # compute and print metrics
    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    # Note: image score isn't good because some images are a priori more likely
    # This can be fixed by using some reference images and then normalizing to get p(caption|image)
    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    # Note: not good because image score is bad
    def group_correct(result):
        return image_correct(result) and text_correct(result)

    def conf_interval(p, n, z=1.96):
        return z * np.sqrt(p * (1 - p) / n)

    correct_counts = dict(Overall=Counter(), Object=Counter(), Relation=Counter(), Both=Counter())
    denominators = Counter()

    for tag, result in zip(all_tags, all_scores):
        for label in ['Overall', tag]:
            correct_counts[label]['text'] += 1 if text_correct(result) else 0
            correct_counts[label]['image'] += 1 if image_correct(result) else 0
            correct_counts[label]['group'] += 1 if group_correct(result) else 0
            denominators[label] += 1

    for tag in correct_counts.keys():
        for metric in correct_counts[tag].keys():
            p = correct_counts[tag][metric] / denominators[tag]
            print(f'{tag} {metric} score: {100 * p : .1f} +/- {100 * conf_interval(p, denominators[tag]) : .1f}')


if __name__ == '__main__':
    main()
