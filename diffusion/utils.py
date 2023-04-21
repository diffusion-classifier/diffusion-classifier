from datetime import datetime
from functools import lru_cache
import json
import os
import os.path as osp
import torch
from PIL import Image

DATASET_ROOT = os.getenv('DATASET_ROOT', '/datasets')
LOG_DIR = os.getenv('LOG_DIR', 'data')
TOKEN_PATH = os.getenv('TOKEN_PATH', osp.expanduser('~/hf_token.txt'))
HDD_ROOT = os.getenv('HDD_ROOT', '')  # should point to the HDD path on each machine
# it is stored in the current directory
TEMPLATE_JSON_PATH = os.path.join(os.path.dirname(__file__), 'templates.json')


def save_latent(vae, latent, path, scaling=1 / 0.18125):
    # scale and decode the image latents with vae
    scaled_latents = scaling * latent

    with torch.no_grad():
        image = vae.decode(scaled_latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    img = pil_images[0]
    img.save(path)


@lru_cache  # same datestr on different calls
def get_datetimestr():
    # only go to 3 ms digits
    return datetime.now().strftime("%Y.%m.%d_%H.%M.%S")


def get_formatstr(n):
    # get the format string that pads 0s to the left of a number, which is at most n
    digits = 0
    while n > 0:
        digits += 1
        n //= 10
    return f"{{:0{digits}d}}"


def get_classes_templates(dataset) -> tuple:
    """Get a template for the text prompt.

    Args:
        dataset: dataset name

    Returns:
        template: template for the text prompt
    """
    with open(TEMPLATE_JSON_PATH, 'r') as f:
        all_templates = json.load(f)

    if dataset not in all_templates:
        raise NotImplementedError(f"Dataset {dataset} not implemented. Only {list(all_templates.keys())} are supported.")
    entry = all_templates[dataset]

    if "classes" not in entry:
        raise ValueError(f"Dataset {dataset} does not have a `classes` entry.")
    if "templates" not in entry:
        raise ValueError(f"Dataset {dataset} does not have a `templates` entry.")

    classes_dict, templates = entry["classes"], entry["templates"]

    # always return a dict of class_key: [class_names...]
    if isinstance(classes_dict, list):
        classes_dict = {c: [c] for c in classes_dict}

    return classes_dict, templates
