import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, \
    EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer


def get_sd_model(args):
    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float16':
        dtype = torch.float16
    else:
        raise NotImplementedError

    if args.version == '1-4':
        vae = AutoencoderKL.from_pretrained(f"CompVis/stable-diffusion-v{args.version}",
                                            subfolder="vae", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        unet = UNet2DConditionModel.from_pretrained(f"CompVis/stable-diffusion-v{args.version}",
                                                    subfolder="unet", torch_dtype=dtype)
        scheduler_config = get_scheduler_config(args)
        scheduler = DDPMScheduler(num_train_timesteps=scheduler_config['num_train_timesteps'],
                                  beta_start=scheduler_config['beta_start'],
                                  beta_end=scheduler_config['beta_end'],
                                  beta_schedule=scheduler_config['beta_schedule'])
    elif args.version == '2-1':
        model_id = "stabilityai/stable-diffusion-2-1-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype)
        pipe.enable_xformers_memory_efficient_attention()
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
    else:
        raise NotImplementedError
    return vae, tokenizer, text_encoder, unet, scheduler


def get_scheduler_config(args):
    assert args.version in {'1-4', '2-1'}
    if args.version == '1-4':
        config = {
            "_class_name": "PNDMScheduler",
            "_diffusers_version": "0.7.0.dev0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
            "clip_sample": False
        }
    elif args.version == '2-1':
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
