import logging
import numpy as np
from torchvision.datasets import FGVCAircraft

from fgvc.datasets.aug_wrapper_dataset import AugWrapperDataset


class Planes(AugWrapperDataset, FGVCAircraft):
    def __init__(self, root='/mnt/raid/home/user_name/datasets/FGVC-Aircraft', 
                 split='train', transform=None, target_transform=None, download=False, train_sample_ratio=1.0,
                 aug_json=None, aug_sample_ratio=None, limit_aug_per_image=None, few_shot=None, print_func=logging.info):
        
        # First, get the needed vars: self._image_files, self._labels, self.num_classes, self.dataset_name
        # in this case FGVCAircraft is already implemented, so we initialize it and get self._image_files, self._labels
        FGVCAircraft.__init__(self, root=root, split=split, annotation_level='variant', transform=transform, target_transform=target_transform, download=download)
        self.dataset_name = "planes"
        self.num_classes = len(np.unique(self._labels))

        # Then initialize AugWrapperDataset
        AugWrapperDataset.__init__(self, root=root, split=split, transform=transform, target_transform=target_transform, 
                                   train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, 
                                   limit_aug_per_image=limit_aug_per_image, few_shot=few_shot, print_func=print_func)
        
