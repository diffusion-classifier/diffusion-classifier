import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, \
    EulerDiscreteScheduler

MODEL_IDS = {
    '1-1': "CompVis/stable-diffusion-v1-1",
    '1-2': "CompVis/stable-diffusion-v1-2",
    '1-3': "CompVis/stable-diffusion-v1-3",
    '1-4': "CompVis/stable-diffusion-v1-4",
    '1-5': "runwayml/stable-diffusion-v1-5",
    '2-0': "stabilityai/stable-diffusion-2-base",
    '2-1': "stabilityai/stable-diffusion-2-1-base"
}


def get_sd_model(args):
    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float16':
        dtype = torch.float16
    else:
        raise NotImplementedError

    assert args.version in MODEL_IDS.keys()
    model_id = MODEL_IDS[args.version]
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype)
    pipe.enable_xformers_memory_efficient_attention()
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    return vae, tokenizer, text_encoder, unet, scheduler


def get_scheduler_config(args):
    if args.version in {'1-1', '1-2', '1-3', '1-4', '1-5'}:
        config = {
            "_class_name": "EulerDiscreteScheduler",
            "_diffusers_version": "0.14.0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "interpolation_type": "linear",
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None
        }
    elif args.version in {'2-0', '2-1'}:
        config = {
            "_class_name": "EulerDiscreteScheduler",
            "_diffusers_version": "0.10.2",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,  # todo
            "trained_betas": None
        }
    else:
        raise NotImplementedError

    return config
