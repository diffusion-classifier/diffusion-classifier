import argparse
import torch


def save_noise():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=64, choices=(256, 512),
                        help='Image size that noise corresponds to.')
    args = parser.parse_args()
    latent_size = args.img_size // 8
    torch.manual_seed(42)
    noise = torch.randn(1024, 4, latent_size, latent_size)

    fname = f'noise_{args.img_size}.pt'
    torch.save(noise, fname)
    print(f"Noise saved to {fname}.")



if __name__ == '__main__':
    save_noise()
