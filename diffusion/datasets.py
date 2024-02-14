import glob
import json
import os
import os.path as osp
import torch
from torchvision import datasets
from diffusion.utils import DATASET_ROOT, get_classes_templates
from diffusion.dataset.objectnet import ObjectNetBase
from diffusion.dataset.imagenet_classnames import get_classnames
from imagenetv2_pytorch import ImageNetV2Dataset
from PIL import Image

IMAGENET_A_CLASSES = [
    6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107, 108,
    110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307,
    308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386,
    397, 400, 401, 402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483,
    486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606,
    607, 609, 614, 626, 627, 640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758,
    763, 765, 768, 773, 774, 776, 779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845,
    847, 850, 859, 862, 870, 879, 880, 888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954,
    956, 957, 959, 971, 972, 980, 981, 984, 986, 987, 988
]

OBJECTNET_CLASSES = [
    409, 412, 414, 418, 419, 423, 434, 440, 444, 446, 455, 457, 462, 463, 470, 473, 479, 487, 499, 504, 507, 508, 518,
    530, 531, 533, 539, 543, 545, 549, 550, 559, 560, 563, 567, 578, 587, 588, 589, 601, 606, 608, 610, 618, 619, 620,
    623, 626, 629, 630, 632, 644, 647, 651, 655, 658, 659, 664, 671, 673, 677, 679, 689, 694, 695, 696, 700, 703, 720,
    721, 725, 728, 729, 731, 732, 737, 740, 742, 749, 752, 759, 761, 765, 769, 770, 772, 773, 774, 778, 783, 790, 792,
    797, 804, 806, 809, 811, 813, 828, 834, 837, 841, 842, 846, 849, 850, 851, 859, 868, 879, 882, 883, 893, 898, 902,
    906, 907, 909, 923, 930, 950, 951, 954, 968, 999
]

with open('diffusion/imagenet_class_index.json', 'r') as f:
    IMAGENET_CLASS_INDEX = json.load(f)
FOLDER_TO_CLASS = {folder: int(i) for i, (folder, _) in IMAGENET_CLASS_INDEX.items()}


class MNIST(datasets.MNIST):
    """Simple subclass to override the property"""
    class_to_idx = {str(i): i for i in range(10)}


class ImageNetV2(ImageNetV2Dataset):
    def __init__(
            self,
            variant="matched-frequency",
            root=DATASET_ROOT,
            transform=None,
    ):
        super(ImageNetV2, self).__init__(variant=variant,
                                         location=root,
                                         transform=transform)
        self.fnames.sort()


class ImageNetA(torch.utils.data.Dataset):
    def __init__(
            self,
            root=DATASET_ROOT,
            transform=lambda x: x,
    ):
        super(ImageNetA, self).__init__()
        self.root = os.path.join(root, 'imagenet-a')
        self._images = glob.glob(self.root + '/*/*')
        self._images.sort()
        self.transform = transform
        assert len(self._images) == 7500, f"Expected {len(self)} images, found {len(self._images)}"

    def __len__(self):
        return 7500

    def path_to_cls(self, path):
        folder = path.split('/')[-2]
        return FOLDER_TO_CLASS[folder]

    def __getitem__(self, index):
        filepath = self._images[index]
        img, label = Image.open(filepath).convert("RGB"), self.path_to_cls(filepath)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_target_dataset(name: str, train=False, transform=None, target_transform=None):
    """Get the torchvision dataset that we want to use.
    If the dataset doesn't have a class_to_idx attribute, we add it.
    Also add a file-to-class map for evaluation
    """

    if name == "cifar10":
        dataset = datasets.CIFAR10(root=DATASET_ROOT, train=train, transform=transform,
                                   target_transform=target_transform, download=True)
    elif name == "stl10":
        dataset = datasets.STL10(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                 target_transform=target_transform, download=True)
        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}
    elif name == "pets":
        dataset = datasets.OxfordIIITPet(root=DATASET_ROOT, split="trainval" if train else "test", transform=transform,
                                         target_transform=target_transform, download=True)

        # lower case every key in the class_to_idx
        dataset.class_to_idx = {k.lower(): v for k, v in dataset.class_to_idx.items()}

        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._images, dataset._labels)}
    elif name == "flowers":
        dataset = datasets.Flowers102(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                      target_transform=target_transform, download=True)
        classes = list(get_classes_templates('flowers')[0].keys())  # in correct order
        dataset.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._image_files, dataset._labels)}
    elif name == "aircraft":
        dataset = datasets.FGVCAircraft(root=DATASET_ROOT, split="trainval" if train else "test", transform=transform,
                                        target_transform=target_transform, download=True)

        # replace backslash with underscore -> need to be dirs
        dataset.class_to_idx = {
            k.replace('/', '_'): v
            for k, v in dataset.class_to_idx.items()
        }

        dataset.file_to_class = {
            fn.split("/")[-1].split(".")[0]: lab
            for fn, lab in zip(dataset._image_files, dataset._labels)
        }
        # dataset.file_to_class = {
        #     fn.split("/")[-1].split(".")[0]: lab
        #     for fn, lab in zip(dataset._image_files, dataset._labels)
        # }

    elif name == "food":
        dataset = datasets.Food101(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                   target_transform=target_transform, download=True)
        dataset.file_to_class = {
            f.name.split(".")[0]: dataset.class_to_idx[f.parents[0].name]
            for f in dataset._image_files
        }
    elif name == "eurosat":
        if train:
            raise ValueError("EuroSAT does not have a train split.")
        dataset = datasets.EuroSAT(root=DATASET_ROOT, transform=transform, target_transform=target_transform,
                                   download=True)
    elif name == 'imagenet':
        assert not train
        dataset = datasets.ImageFolder(root=osp.join(DATASET_ROOT, 'imagenet/val'),
                                       transform=transform,
                                       target_transform=target_transform)
        dataset.class_to_idx = None
        dataset.classes = get_classnames('openai')
        dataset.file_to_class = None
    elif name == 'objectnet':
        base = ObjectNetBase(transform, DATASET_ROOT)
        dataset = base.get_test_dataset()
        dataset.class_to_idx = dataset.label_map
        dataset.file_to_class = None  # todo
    elif name == 'imagenetv2':
        assert not train
        dataset = ImageNetV2(root=DATASET_ROOT, transform=transform)
        dataset.file_to_class = None
        dataset.class_to_idx = None
    elif name == 'imagenetA':
        dataset = ImageNetA(root=DATASET_ROOT, transform=transform)
        dataset.file_to_class = None
        dataset.class_to_idx = None
    elif name == "caltech101":
        if train:
            raise ValueError("Caltech101 does not have a train split.")
        dataset = datasets.Caltech101(root=DATASET_ROOT, target_type="category", transform=transform,
                                      target_transform=target_transform, download=True)

        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.categories)}
        dataset.file_to_class = {str(idx): dataset.y[idx] for idx in range(len(dataset))}
    elif name == "mnist":
        dataset = MNIST(root=DATASET_ROOT, train=train, transform=transform, target_transform=target_transform,
                        download=True)
    else:
        raise ValueError(f"Dataset {name} not supported.")

    if name in {'mnist', 'cifar10', 'stl10', 'aircraft'}:
        dataset.file_to_class = {
            str(idx): dataset[idx][1]
            for idx in range(len(dataset))
        }

    assert hasattr(dataset, "class_to_idx"), f"Dataset {name} does not have a class_to_idx attribute."
    assert hasattr(dataset, "file_to_class"), f"Dataset {name} does not have a file_to_class attribute."
    return dataset
