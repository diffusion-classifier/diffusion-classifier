<div align="center">

<!-- TITLE -->
# **Your Diffusion Model is Secretly a Zero-Shot Classifier**

[![arXiv](https://img.shields.io/badge/cs.LG-arXiv:2303.16203-b31b1b.svg)](https://arxiv.org/abs/2303.16203)
[![Website](https://img.shields.io/badge/🌎-Website-blue.svg)](http://diffusion-classifier.github.io)
</div>

This is the official implementation of the paper [Your Diffusion Model is Secretly a Zero-Shot Classifier](https://arxiv.org/abs/2303.16203) by Alexander Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, and Deepak Pathak. 
<!-- DESCRIPTION -->
## Abstract

The recent wave of large-scale text-to-image diffusion models has dramatically increased our text-based image generation abilities. These models can generate realistic images for a staggering variety of prompts and exhibit impressive compositional generalization abilities. Almost all use cases thus far have solely focused on sampling; however, diffusion models can also provide conditional density estimates, which are useful for tasks beyond image generation.

In this paper, we show that the density estimates from large-scale text-to-image diffusion models like Stable Diffusion can be leveraged to perform zero-shot classification *without any additional training*. Our generative approach to classification, which we call **Diffusion Classifier**, attains strong results on a variety of benchmarks and outperforms alternative methods of extracting knowledge from diffusion models. We also find that our diffusion-based approach has stronger multimodal relational reasoning abilities than competing contrastive approaches.

Finally, we use Diffusion Classifier to extract standard classifiers from class-conditional diffusion models trained on ImageNet. Even though these diffusion models are trained with weak augmentations and no regularization, we find that they approach the performance of SOTA discriminative ImageNet classifiers. Overall, our strong generalization and robustness results represent an encouraging step toward using generative over discriminative models for downstream tasks.

## Code

### Installation 
Create a conda environment with the following command:
```bash
conda env create -f environment.yml
```

### Zero-shot Classification with Stable Diffusion

```bash
python eval_prob_adaptive.py --dataset cifar10 --split test --n_trials 1 \
  --to_keep 10 5 1 --n_samples 50 100 500 --loss l1 \
  --prompt_path prompts/cifar10_prompts.csv
```
This command reads potential prompts from a csv file and evaluates the epsilon prediction loss for each prompt using Stable Diffusion.
This should work on a variety of GPUs, from as small as a 2080Ti or 3080 to as large as a 3090 or A6000. 
Losses are saved separately for each test image in the log directory. For the command above, the log directory is `data/cifar10/v2-1_1trials_10_5_1keep_50_100_500samples_l1`. Accuracy can be computed by running: 
```bash
python scripts/print_acc.py data/cifar10/v2-1_1trials_10_5_1keep_50_100_500samples_l1
```

Commands to run Diffusion Classifier on each dataset are [here](commands.md). 
If evaluation on your use case is taking too long, there are a few options: 
1. Parallelize evaluation across multiple workers. Try using the `--n_workers` and `--worker_idx` flags.
2. Play around with the evaluation strategy (e.g. `--n_samples` and `--to_keep`).
3. Evaluate on a smaller subset of the dataset. Saving a npy array of test set indices and using the `--subset_path` flag can be useful for this.

### Evaluating on your own dataset
1. Create a csv file with the prompts that you want to evaluate, making sure to match up the correct prompts with the correct class labels. See `scripts/write_cifar10_prompts.py` for an example.
2. Run the command above, changing the `--dataset` and `--prompt_path` flags to match your use case.
3. Play around with the evaluation strategy on a small subset of the dataset to reduce evaluation time.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@misc{li2023diffusion,
      title={Your Diffusion Model is Secretly a Zero-Shot Classifier}, 
      author={Alexander C. Li and Mihir Prabhudesai and Shivam Duggal and Ellis Brown and Deepak Pathak},
      year={2023},
      eprint={2303.16203},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
