import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import FGVCAircraft

from fgvc.datasets.aug_wrapper_dataset import AugWrapperDataset


def get_counts(labels):
    values, counts = np.unique(labels, return_counts=True)
    sorted_tuples = zip(*sorted(zip(values, counts))) # this just ensures we are getting the counts in the sorted order of the keys
    values, counts = [ list(tuple) for tuple in  sorted_tuples]
    fracs   = 1 / torch.Tensor(counts)
    return fracs / torch.max(fracs)


class PlanesBiased(AugWrapperDataset, FGVCAircraft):
    def __init__(self, root='/mnt/raid/home/eyal_michaeli/datasets/FGVC-Aircraft', 
                 split='train', transform=None, target_transform=None, download=False, train_sample_ratio=1.0,
                 aug_json=None, aug_sample_ratio=None, limit_aug_per_image=None, few_shot=None, print_func=logging.info):
        # some code taken fromm ALIA: https://github.com/lisadunlap/ALIA/blob/6e2c00f1f3ecfe0a8784ee4bd71ead5fa3bc6ad4/datasets/planes.py

        # First, get the needed vars: self._image_files, self._labels, self.num_classes, self.dataset_name
        # in this case FGVCAircraft is already implemented, so we initialize it and get self._image_files, self._labels
        FGVCAircraft.__init__(self, root=root, split=split, annotation_level='variant', transform=transform, target_transform=target_transform, download=download)
        self.dataset_name = "planes-biased"
        self.split = split
        self.images_path = str(Path(root) / "fgvc-aircraft-2013b/data/images")
        csv_file = Path(__file__).parent.parent / "datasets_files/aircraft_biased_dataset/alia_cotextual_bias_split.csv"
        self.df = pd.read_csv(csv_file)

        if self.split in ['train', 'test']:
            self.df = self.df[self.df['Split'] == split] if split != 'extra' else self.df[self.df['Split'] == 'val']
        if self.split == 'val':
            self.df = self.df[self.df['Split'] == split][::2]
        if self.split == 'extra': # remove unbiased examples
            # talk half of val set and move it to train
            self.df = self.df[self.df['Split'] == 'val'][1::2]
            # self.df = pd.concat([self.df[self.df['Split'] == 'train'], extra_df])

        self._image_files = np.array([os.path.join(self.images_path, Path(f).name) for f in self.df['Filename']])
        self._labels = np.array(self.df['Label'])
        # from ALIA code, not needed:
        # self.targets = self._labels
        # self.domain_classes = sorted(np.unique(self.df['Ground']))
        # self.domains = np.array([self.domain_classes.index(d) for d in self.df['Ground']])
        # self.groups = np.array(self.df['Group'])
        # self.class_weights = get_counts(self._labels)
        # self.samples = list(zip(self._image_files, self._labels))

        self.class_names = ['airbus', 'boeing']
        # self.group_names = GROUP_NAMES_AIR_GROUND 
        self.classes = ['airbus', 'boeing']

        self.num_classes = len(np.unique(self._labels))

        # Then initialize AugWrapperDataset
        AugWrapperDataset.__init__(self, root=root, split=split, transform=transform, target_transform=target_transform, 
                                   train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, 
                                   limit_aug_per_image=limit_aug_per_image, print_func=print_func)
        

