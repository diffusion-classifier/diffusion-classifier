# Commands for Zero-Shot Classification with Stable Diffusion 
### Food101
```bash
python eval_prob_adaptive.py --dataset food --split test --n_trials 1 \
  --to_keep 20 10 5 1 --n_samples 20 50 100 500 \
  --prompt_path prompts/food_prompts.csv
```

### CIFAR-10
```bash
python eval_prob_adaptive.py --dataset cifar10 --split test --n_trials 1 \
  --to_keep 5 1 --n_samples 50 500 --loss l1 \
  --prompt_path prompts/cifar10_prompts.csv
```

### FGVC Aircraft
```bash
python eval_prob_adaptive.py --dataset aircraft --split test --n_trials 1 \
  --to_keep 20 10 5 1 --n_samples 20 50 100 500 \
  --prompt_path prompts/aircraft_prompts.csv
```

### Oxford-IIIT Pets
```bash
python eval_prob_adaptive.py --dataset pets --split test --n_trials 1 \
  --to_keep 5 1 --n_samples 25 250 --loss l1 \
  --prompt_path prompts/pets_prompts.csv
```

### Flowers102
```bash
python eval_prob_adaptive.py --dataset flowers --split test --n_trials 1 \
  --to_keep 20 10 5 1 --n_samples 20 50 100 500 --loss l1 \
  --prompt_path prompts/flowers_prompts.csv
```

### STL-10
```bash
python eval_prob_adaptive.py --dataset stl10 --split test --n_trials 1 \
  --to_keep 5 1 --n_samples 100 500 --loss l1 \
  --prompt_path prompts/stl10_prompts.csv
```

### ImageNet
```bash
python eval_prob_adaptive.py --dataset imagenet --split test --n_trials 1 \
  --to_keep 500 50 10 1 --n_samples 50 100 500 1000 \
  --prompt_path prompts/imagenet_prompts.csv
```

Note: for computational reasons, we evaluated on 4 images per class (4000 test images total).

### ObjectNet
```bash
python eval_prob_adaptive.py --dataset objectnet --split test --n_trials 1 \
  --to_keep 25 10 5 1 --n_samples 50 100 500 1000 \
  --prompt_path prompts/objectnet_prompts.csv
```

## Commands for Standard ImageNet Classification with DiT
If you'd like to run with 512x512 DiT, add `--image-size 512` and change `noise_256.pt` to `noise_512.pt`.
### ImageNet
Run 1000 separate times, with CLS from 0-999
```bash
python eval_prob_dit.py  --dataset imagenet --split test \
  --noise_path noise_256.pt --randomize_noise \
  --batch_size 32 --cls CLS --t_interval 4 --extra dit256 --save_vb
```
### ImageNet-V2
Run 1000 separate times, with CLS from 0-999
```bash
python eval_prob_dit.py  --dataset imagenetv2 --split test \
  --noise_path noise_256.pt --randomize_noise \
  --batch_size 32 --cls CLS --t_interval 4 --extra dit256 --save_vb
```
### ImageNet-A
ImageNet-A only has 200 classes, so run this command with CLS from `IMAGENET_A_CLASSES` in [datasets.py](diffusion/datasets.py).
```bash
python eval_prob_dit.py  --dataset imagenetA --split test \
  --noise_path noise_256.pt --randomize_noise \
  --batch_size 32 --cls CLS --t_interval 4 --extra dit256 --save_vb
```
### ObjectNet
ObjectNet only has 125 classes, so run this command with CLS from `OBJECTNET_CLASSES` in [datasets.py](diffusion/datasets.py).
```bash
python eval_prob_dit.py  --dataset objectnet --split test \
  --noise_path noise_256.pt --randomize_noise \
  --batch_size 32 --cls CLS --t_interval 4 --extra dit256 --save_vb
```
