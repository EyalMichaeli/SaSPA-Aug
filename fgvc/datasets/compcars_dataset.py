""" Stanford Cars (Car) Dataset """
import json
import logging
import os
from pathlib import Path
import pdb
import random
from typing import Callable, Optional
import warnings
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from fgvc.datasets.aug_wrapper_dataset import AugWrapperDataset


ROOT = Path("").parent.parent / "data/compcars/part"

class CompCars(AugWrapperDataset, Dataset):
    def __init__(self, root: str = ROOT, split: str = "train", 
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, train_sample_ratio: float = 1.0, 
                 aug_json: str = None, aug_sample_ratio: float = None, limit_aug_per_image: int = None, dataset_type="parts", 
                 print_func=logging.info, few_shot=None, create_val_split=False):
        """
        the label is a string of car_make/car_model/year, path like, all numbers. e.g: 1/1/1.jpg, 10/5/2012, 1/1001,2011
        """
        assert split in ['train', 'val', 'test']
        split_to_load = 'train' if split == 'val' else split  
        
        Dataset.__init__(self)
        # First, get the needed vars: self._image_files, self._labels, self.num_classes, self.dataset_name
        self.root = root
        self.dataset_type = dataset_type
        if self.dataset_type == "parts":
            split_csv_file = Path("").parent.parent / "fgvc/datasets_files/compcars-parts" / f"{split_to_load}.csv"
        else:
            raise NotImplementedError(f"Dataset type {self.dataset_type} is not implemented. Supported types: parts.")

        # load the csv file with paths, labels
        self._labels = []
        self.label_to_class_id_map = {}
        self.class_id_to_car_make_map = {}
        self.class_id_to_car_model_map = {}
        self._image_files = []
        with open(split_csv_file, 'r') as f:
            for line in f:
                path, label = line.strip().split(',')
                self._image_files.append(str(Path(root) / path))
                self._labels.append(label)
        
        all_unique_labels_sorted = sorted(list(set(self._labels)))
        # create a dict of label: class_id
        self.label_to_class_id_map = {label: i for i, label in enumerate(all_unique_labels_sorted)}
        self._labels = [self.label_to_class_id_map[label] for label in self._labels]
        self.class_id_to_label_map = {v: k for k, v in self.label_to_class_id_map.items()}
        # create a dict of class_id: car_make/car_model/car_year
        for class_id, label in self.class_id_to_label_map.items():
            if self.dataset_type == "parts":
                car_make, car_model = label.split('/')
            else:
                car_make, car_model, car_year = label.split('/')
            self.class_id_to_car_make_map[class_id] = car_make
            self.class_id_to_car_model_map[class_id] = car_model
        
        # if split is val, load the val txt file: 
        if split in ['train', 'val']:
            file_path = os.path.join(str(Path(__file__).parent.parent), 'datasets_files', 'compcars_parts_val.txt')
            with open(file_path, 'r') as f:
                val_image_files = [line.strip() for line in f.readlines()]
            
            new_image_files = []
            new_labels = []
            for image_file, label in zip(self._image_files, self._labels):
                relevant_image_file = str(Path(*Path(image_file).parts[-5:]))
                if (split == "val" and relevant_image_file in val_image_files) or (split == "train" and relevant_image_file not in val_image_files):
                    new_image_files.append(image_file)
                    new_labels.append(label)
            self._image_files = new_image_files
            self._labels = new_labels

        self.classes = list(set(self._labels))
        # get a dict of label: num samples for the plot of num samples per class vs. class accuracy
        # self.label_to_num_samples = {label: self._labels.count(label) for label in self.classes}
        self.num_classes = len(set(self.classes))
        self.dataset_name = "compcars"

        # Then initialize AugWrapperDataset
        AugWrapperDataset.__init__(self, root=root, split=split, transform=transform, target_transform=target_transform, 
                                   train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, 
                                   limit_aug_per_image=limit_aug_per_image, print_func=print_func, create_val_split=create_val_split, few_shot=few_shot)

if __name__ == '__main__':
    # small test
    from fgvc.util import get_transform
    transform = get_transform(resize=(224, 224), phase='train')

    ds = CompCars(split='test', transform=transform, print_func=print)
    print(len(ds))
    for i in range(0, 10):
        image, label = ds[i]
        print(image.shape, label)
