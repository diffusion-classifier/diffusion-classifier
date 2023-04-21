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
  --to_keep 10 5 1 --n_samples 50 100 500 --loss l1 \
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
  --to_keep 5 1 --n_samples 100 1000 --loss l1 \
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