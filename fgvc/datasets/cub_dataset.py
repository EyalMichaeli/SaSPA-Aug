""" CUB-200-2011 (Bird) Dataset"""
import json
import logging
import os
from pathlib import Path
import random
import warnings
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from typing import Callable, Optional

from fgvc.datasets.aug_wrapper_dataset import AugWrapperDataset


class CUB(AugWrapperDataset, Dataset):
    """
    # Description:
    CUB 200-2011 Dataset
    https://paperswithcode.com/dataset/cub-200-2011
    """

    def __init__(self, root: str = "/mnt/raid/home/eyal_michaeli/datasets/CUB/CUB_200_2011", split: str = "train", 
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, 
                 train_sample_ratio: float = 1.0, aug_json: str = None, aug_sample_ratio: float = None, 
                 limit_aug_per_image: int = None, print_func=logging.info, few_shot=None, create_val_split=False):
        self.root = root
        self.split = split  # val is the same as test

        # First, get the needed vars: self._image_files, self._labels, self.num_classes, self.dataset_name
        self._image_files = []
        self._labels = []
        self.num_classes = 200
        self.dataset_name = "cub"
        self.is_train = "train" in split

        image_path = {}
        image_label = {}

        # get image path from images.txt
        with open(os.path.join(root, 'images.txt')) as f:
            for line in f.readlines():
                id, path = line.strip().split(' ')
                image_path[id] = str(Path(root) / 'images' / path)

        # get image label from image_class_labels.txt
        with open(os.path.join(root, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                id, label = line.strip().split(' ')
                image_label[id] = int(label) - 1 # count begin from zero

        # get train/test image id from train_test_split.txt
        with open(os.path.join(root, 'train_test_split.txt')) as f:
            for line in f.readlines():
                image_id, is_training_image = line.strip().split(' ')
                is_training_image = int(is_training_image)

                if self.split in ('train', 'val') and is_training_image:
                    self._image_files.append(image_path[image_id])
                    self._labels.append(image_label[image_id])
                if self.split in ['test'] and not is_training_image:
                    self._image_files.append(image_path[image_id])
                    self._labels.append(image_label[image_id])

        # if split is val, load the val txt file: 
        if split in ['train', 'val']:
            file_path = os.path.join(str(Path(__file__).parent.parent), 'datasets_files', 'cub_val.txt')
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

    ds = CUB(split='test', transform=transform)
    print(len(ds))
    for i in range(0, 10):
        image, label = ds[i]
        print(image.shape, label)
