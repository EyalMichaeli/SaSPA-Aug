
import json
import logging
import os
from pathlib import Path
import pdb
import random
import warnings
from PIL import Image
import numpy as np
from torchvision.datasets import DTD

from fgvc.datasets.aug_wrapper_dataset import AugWrapperDataset


ROOT = Path("").parent.parent / 'data/DTD'

class DTDataset(AugWrapperDataset, DTD):
    """
    DTD dataset.
    https://www.robots.ox.ac.uk/~vgg/data/dtd/
    https://paperswithcode.com/sota/image-classification-on-dtd
    """
    def __init__(self, root=ROOT, 
                 split='train', transform=None, target_transform=None, download=True, train_sample_ratio=1.0,
                 aug_json=None, aug_sample_ratio: float = None, limit_aug_per_image: int = None, print_func=logging.info, few_shot=None):
        DTD.__init__(self, root=root, split=split, transform=transform, target_transform=target_transform, download=download)
        # First, get the needed vars: self._image_files, self._labels, self.num_classes, self.dataset_name
        # in this case, DTD is ready in pytorch format, so we can just use it
        self.num_classes = len(np.unique(self._labels))
        self.dataset_name = "dtd"

        # Then initialize AugWrapperDataset
        AugWrapperDataset.__init__(self, root=root, split=split, transform=transform, target_transform=target_transform, 
                                   train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, 
                                   limit_aug_per_image=limit_aug_per_image, print_func=print_func, few_shot=few_shot)

if __name__ == '__main__':
    # small test
    from fgvc.util import get_transform
    transform = get_transform(resize=(224, 224), phase='train')

    ds = DTDataset(split='train', transform=transform)
    print(len(ds))
    for i in range(0, 100):
        image, label = ds[i]
        print(image.shape, label)
