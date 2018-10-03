# -*- coding: utf-8 -*-

from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image


class ImageFolderDataset(Dataset):
    """A variant of torchvision.datasets.ImageFolder which drops support for
    target loading, i.e. this only loads images without labels.

    Arguments:
        root (str): The root folder which contains a folder per each split.
        split (str): A subfolder that should exist under ``root`` containing
            images for a specific split.
        resize (int, optional): An optional integer to be given to
            ``torchvision.transforms.Resize``. Default: ``None``.
        crop (int, optional): An optional integer to be given to
            ``torchvision.transforms.CenterCrop``. Default: ``None``.
    """
    def __init__(self, root, split, resize=None, crop=None):
        self.split = split
        self.root = Path(root).expanduser().resolve() / self.split

        # Image list in dataset order
        self.index = self.root / 'index.txt'

        _transforms = []
        if resize is not None:
            _transforms.append(transforms.Resize(resize))
        if crop is not None:
            _transforms.append(transforms.CenterCrop(crop))
        _transforms.append(transforms.ToTensor())
        _transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(_transforms)

        if not self.index.exists():
            raise(RuntimeError(
                "index.txt does not exist in {}".format(self.root)))

        self.image_files = []
        with self.index.open() as f:
            for fname in f:
                fname = self.root / fname.strip()
                assert fname.exists(), "{} does not exist.".format(fname)
                self.image_files.append(str(fname))

    def read_image(self, fname):
        with open(fname, 'rb') as f:
            img = Image.open(f).convert('RGB')
            return self.transform(img)

    def __getitem__(self, idx):
        return self.read_image(self.image_files[idx])

    def __len__(self):
        return len(self.image_files)
