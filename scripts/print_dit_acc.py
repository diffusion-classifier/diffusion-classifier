import argparse
import os
import os.path as osp
import torch
from diffusion.datasets import get_target_dataset, IMAGENET_A_CLASSES
from tqdm import tqdm
import multiprocessing as mp

N_WORKERS = 16


def filter_imageneta(files, labels):
    # leave only files that whose labels are in IMAGENET_A_CLASSES
    cls_set = set(IMAGENET_A_CLASSES)
    return [f for f in files if labels[int(f.split('.')[0])] in cls_set]


def get_imageneta_files(args, labels, test_subdir=True):
    # get subset of folders/files that are in ImageNet-A
    subdirs = [f'DiT256x256_cls{cls}_t4_1trials_randnoise' for cls in IMAGENET_A_CLASSES]

    files_per_prompt = [
        set(os.listdir(
            osp.join(args.folder, subdir, "test")
            if test_subdir else osp.join(args.folder, subdir)
        )) for subdir in subdirs
    ]
    shared_files = sorted(set.intersection(*files_per_prompt))
    shared_files = filter_imageneta(shared_files, labels)
    return subdirs, shared_files


def get_shared_files(args, labels, test_subdir=True, imageneta=False):
    if imageneta:
        subdirs, shared_files = get_imageneta_files(args, labels, test_subdir=test_subdir)
    else:
        subdirs = [f for f in os.listdir(args.folder) if osp.isdir(osp.join(args.folder, f))]

        files_per_prompt = [
            set(os.listdir(
                osp.join(args.folder, subdir, "test")
                if test_subdir else osp.join(args.folder, subdir)
            )) for subdir in subdirs
        ]
        shared_files = sorted(set.intersection(*files_per_prompt))
    print(f"Found {len(shared_files)} shared files across all {len(subdirs)} subdirs.")
    return subdirs, shared_files


def cls_from_dir(name):
    chunks = name.split('_')
    for chunk in chunks:
        if chunk.startswith('cls'):
            return int(chunk[3:])
    raise ValueError(f"Could not find class idx in {name}.")


def is_correct(labels, folder, subdirs, file, test_subdir=True, idx_map=None):
    gt_label = labels[int(file.split('.')[0])]
    scores_labels = [[], [], [], []]
    for subdir in subdirs:
        label = cls_from_dir(subdir)
        error_file = os.path.join(folder, subdir, "test", file) if \
            test_subdir else os.path.join(folder, subdir, file)
        if not os.path.exists(error_file):
            raise FileNotFoundError(f"File {error_file} does not exist.")
        try:
            error = torch.load(error_file).mean(dim=(0, 1))
        except:
            raise ValueError(error_file)
        scores_labels[0].append((error[0], label))
        scores_labels[1].append((error[1], label))
        scores_labels[2].append((error[2], label))
        scores_labels[3].append((error[0] + error[2], label))
    result = []
    for scores in scores_labels:
        sorted_scores = sorted(scores, key=lambda x: x[0])
        pred_label = sorted_scores[0][1]
        if idx_map is not None:
            pred_label = idx_map[pred_label]
        result.append(int(pred_label == gt_label))
    return result


def is_correct_wrapper(args):
    return is_correct(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='Folder containing predictions')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'objectnet', 'imagenetv2', 'imagenetA'],
                        help='Dataset to use')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test', 'trainval'], help='Name of split')
    args = parser.parse_args()

    # get labels
    idx_map = None
    test_subdir = True
    dataset = get_target_dataset(args.dataset, train=args.split == 'train')
    if args.dataset == 'imagenet':
        labels = dataset.targets
    elif args.dataset == 'imagenetv2':
        labels = [int(dataset.fnames[i].parent.name) for i in range(len(dataset))]
    elif args.dataset == 'imagenetA':
        labels = [dataset.path_to_cls(path) for path in dataset._images]
    elif args.dataset == 'objectnet':
        labels = [dataset.label_map[osp.basename(osp.dirname(path))] for path, _ in dataset.samples]
        idx_map = dataset.class_idx_map
        test_subdir = False
    else:
        return NotImplementedError

    subdirs, shared_files = get_shared_files(args, labels, test_subdir=test_subdir, imageneta=False)
    with mp.Pool(N_WORKERS) as pool:
        arglist = [(labels, args.folder, subdirs, f, test_subdir, idx_map) for f in shared_files]
        results = list(tqdm(pool.imap(is_correct_wrapper, arglist), total=len(shared_files)))

    results = list(zip(*results))
    names = ['L2', 'L1', 'VLB', 'L2+VLB']
    for i, result in enumerate(results):
        print(f"Accuracy for {names[i]}: {sum(result) / len(result)}")


if __name__ == '__main__':
    main()
