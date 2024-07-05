""" Stanford Cars (Car) Dataset """
import json
import logging
import os
from pathlib import Path
import random
from typing import Callable, Optional
import warnings
from PIL import Image
import numpy as np
from torchvision.datasets import StanfordCars

from fgvc.datasets.aug_wrapper_dataset import AugWrapperDataset

ROOT = "/mnt/raid/home/eyal_michaeli/datasets/"
class Cars(AugWrapperDataset, StanfordCars):
    def __init__(self, root: str = ROOT, split: str = "train", 
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = True, 
                 train_sample_ratio: float = 1.0, aug_json: str = None, aug_sample_ratio: float = None, 
                 limit_aug_per_image: int = None, print_func=logging.info, few_shot=None, create_val_split=False):
        split_to_load = 'train' if split == 'val' else split  
        StanfordCars.__init__(self, root=root, split=split_to_load, transform=transform, target_transform=target_transform, download=download)
        # First, get the needed vars: self._image_files, self._labels, self.num_classes, self.dataset_name
        
        self._image_files = [sapmle[0] for sapmle in self._samples]
        self._labels = [sapmle[1] for sapmle in self._samples]
        self.num_classes = len(set(self.classes))
        self.dataset_name = "cars"

        self.total_num_images_train = len(self._image_files)  # before the sampling, to take the right amount of validation images
        self.all_image_files = self._image_files.copy()

        # if split is val, load the val txt file: 
        if split in ['train', 'val']:
            file_path = os.path.join(str(Path(__file__).parent.parent), 'datasets_files', 'cars_val.txt')
            with open(file_path, 'r') as f:
                val_image_files = [line.strip() for line in f.readlines()]
            
            new_image_files = []
            new_labels = []
            for image_file, label in zip(self._image_files, self._labels):
                if (split == "val" and image_file in val_image_files) or (split == "train" and image_file not in val_image_files):
                    new_image_files.append(image_file)
                    new_labels.append(label)
            self._image_files = new_image_files
            self._labels = new_labels

        self.total_num_images_train = len(self._image_files)  # before the sampling, to take the right amount of val
        self.all_image_files = self._image_files.copy()
        
        # Then initialize AugWrapperDataset
        AugWrapperDataset.__init__(self, root=root, split=split, transform=transform, target_transform=target_transform, 
                                   train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, 
                                   limit_aug_per_image=limit_aug_per_image, print_func=print_func, create_val_split=create_val_split, few_shot=few_shot)

if __name__ == '__main__':
    # small test
    from fgvc.util import get_transform
    transform = get_transform(resize=(224, 224), phase='train')

    ds = Cars(split='test', transform=transform, print_func=print)
    print(len(ds))
    for i in range(0, 10):
        image, label = ds[i]
        print(image.shape, label)
