import json
import logging
from pathlib import Path
import random
from typing import Callable, Optional
import warnings
from PIL import Image
import numpy as np


class AugWrapperDataset:
    def __init__(self, root: str = "", split: str = "train", transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None, train_sample_ratio: float = 1.0, aug_json: str = None, 
                 aug_sample_ratio: float = None, limit_aug_per_image: int = None, few_shot: int = None, print_func=logging.info, create_val_split: bool = False):
        assert root != ""
        assert split in ['train', 'val', 'test']
        assert not (few_shot and train_sample_ratio < 1), "few_shot and train_sample_ratio can't be used together"

        if few_shot is not None:
            create_val_split = False  # few-shot is only for train, so no need for a val split
        ######################################################################################################
        # this 4 vars are needed to be defined in the child class 
        self._image_files: list  # list of image paths
        self._labels: list  # list of labels (class ids)
        self.num_classes: int  # number of classes
        self.dataset_name: str  # name of the dataset (used just for logging)
        ######################################################################################################

        self.print_func = print_func
        self.is_train = 'train' in split
        self.original_data_length = self.__len__()

        if split == "train" and create_val_split:  # create a val split from the train split, in case u want to do K-fold validation
            self.total_num_images_train = len(self._image_files)  # before the sampling, to take the right amount of validation images
            self.all_image_files = self._image_files.copy()  # all train images
            self._all_labels = self._labels.copy()  # all train labels

            num_for_val = int(self.total_num_images_train * 0.33)

            indices_for_val = random.sample(range(self.total_num_images_train), num_for_val)

            # for train, this might be later sampled again according to train_sample_ratio
            self._image_files = [self._image_files[i] for i in range(self.total_num_images_train) if i not in indices_for_val]
            self._labels = [self._labels[i] for i in range(self.total_num_images_train) if i not in indices_for_val]
            
            # for val
            self._val_image_files = [self.all_image_files[i] for i in indices_for_val]
            self._val_labels = [self._all_labels[i] for i in indices_for_val]

        # use only a subset of the images for training, if train_sample_ratio < 1
        if self.is_train and train_sample_ratio < 1:
            self._image_files, self._labels = self.use_subset(train_sample_ratio, self._image_files, self._labels)

        if self.is_train and few_shot is not None:
            self.use_few_shot(few_shot)
        
        self.transform = transform
        self.target_transform = target_transform

        self.print_func(f"DATASET: {self.dataset_name}, SPLIT: {split}")
        self.print_func("LEN DATASET: {}".format(len(self._image_files)))
        self.print_func("NUM CLASSES: {}".format(self.num_classes))

        self.stop_aug = False
        if self.is_train and aug_json and aug_sample_ratio > 0:
            self.init_augmentation(aug_json, aug_sample_ratio, limit_aug_per_image)
        else:
            self.aug_json = None
            self.print_func("Not using DiffusionAug images")    
        
    def use_few_shot(self, k):
        assert self.is_train
        assert k > 0
        self.print_func(f"Using few-shot with {k} shots per class")
        # keep only k images per class
        label_to_image_path = {label: [] for label in self._labels}
        for i, label in enumerate(self._labels):
            label_to_image_path[label].append(self._image_files[i])
        
        selected_images = []
        selected_labels = []
        for label, images in label_to_image_path.items():
            selected_images += images[:k]
            selected_labels += [label] * k

        self._image_files = selected_images
        self._labels = selected_labels
        # assert number of samples is correct:
        assert len(self._image_files) == self.num_classes * k
        self.print_func(f"Using {len(self._image_files)} images for few-shot training")

    
    def use_subset(self, sample_ratio, images_paths, labels):
        assert sample_ratio > 0 and sample_ratio <= 1
        subset_size = int(len(images_paths) * sample_ratio)
        indices_to_take = np.random.choice(len(images_paths), subset_size, replace=False)
        
        self.print_func(f"With ratio {sample_ratio}, using only {subset_size} images for training, out of {len(images_paths)}")
        
        selected_images = np.array(images_paths)[indices_to_take]
        selected_labels = np.array(labels)[indices_to_take]
        
        return list(selected_images), list(selected_labels)


    def init_augmentation(self, aug_json, aug_sample_ratio, limit_aug_per_image):
        self.limit_aug_per_image = limit_aug_per_image
        assert aug_sample_ratio is not None
        assert aug_sample_ratio > 0 and aug_sample_ratio <= 1
        with open(aug_json, 'r') as f:
            self.aug_json = json.load(f)
        # leave only keys that thier values (which is a list) is not empty
        self.aug_json = {k: v[:self.limit_aug_per_image] for k, v in self.aug_json.items() if v}
        assert len(self.aug_json) > 0, "aug_json is empty"

        if self.limit_aug_per_image is not None:  
            self.print_func(f"Using a max of {self.limit_aug_per_image} augmented images per original image")
            # test it:
            assert max([len(v) for v in self.aug_json.values()]) <= self.limit_aug_per_image, "limit_aug_per_image must be >= the number of augmented images per original image"

        self.aug_sample_ratio = aug_sample_ratio
        self.times_used_orig_images = 0
        self.times_used_aug_images = 0

        # if aug ratio is 1: remove samples that don't have an augmented image
        if aug_sample_ratio == 1:
            original_train_length = len(self._image_files)
            image_names_in_json = [Path(img).name for img in self.aug_json]
            indices_with_aug = [i for i, img in enumerate(self._image_files) if Path(img).name in image_names_in_json]
            self._image_files = [img for i, img in enumerate(self._image_files) if i in indices_with_aug]
            self._labels = [label for i, label in enumerate(self._labels) if i in indices_with_aug]
            self.print_func(f"Using only images that have augmented images, {len(self._image_files)} images left out of {original_train_length} original images")
            self.original_data_length = len(self._image_files)

        self.print_func(f"Using augmented images with ratio {aug_sample_ratio}")
        self.print_func(f"There are {len(self.aug_json)} augmented images, out of {self.original_data_length} original images, \n which is {round(len(self.aug_json)/self.original_data_length, 2)*100}% of the original images")
        self.print_func(f"json file: {aug_json}")


    def __len__(self):
        return len(self._image_files)


    def get_aug_image(self, image_path, idx):
        ratio_used_aug = 0
        if random.random() < self.aug_sample_ratio:
            original_image_path = image_path
            aug_img_files = self.aug_json.get(Path(image_path).name, [image_path])  # if image_path is not in aug_json, returns image_path
            aug_img_files = [image_path] if len(aug_img_files) == 0 else aug_img_files  # if image_path key in the json returns an enpty list, use current image_path
            image_path = random.choice(aug_img_files)
            if original_image_path == image_path:  # didn't use augmented image
                self.times_used_orig_images += 1

            else:  # used augmented image
                self.times_used_aug_images += 1
            pass

        else:
            self.times_used_orig_images += 1

        ratio_used_aug = self.times_used_aug_images / (self.times_used_orig_images + self.times_used_aug_images)

        if idx % 100 == 0 and idx > 99 and ratio_used_aug < self.aug_sample_ratio / 3:  # check every 100 iters. e.g, if aug_sample_ratio = 0.3, then warn if ratio is less than 0.1
            warn = f"Using augmented images might be lacking, ratio: {ratio_used_aug:.4f} when it should be around {self.aug_sample_ratio}. This might make sense if a lot were filtered out."
            warnings.warn(warn)
            self.print_func(f"self.times_used_aug_images = {self.times_used_aug_images}, self.times_used_orig_images = {self.times_used_orig_images}")
            
        # every 500 iters, print the ratio of original images to augmented images
        if idx % 1000 == 0:
            self.print_func(f"Used augmented images {(ratio_used_aug*100):.4f}% of the time")
        return image_path


    def __getitem__(self, idx):
        image_path, label = str(self._image_files[idx]), self._labels[idx]

        if self.is_train and self.aug_json and not self.stop_aug:
            image_path = self.get_aug_image(image_path, idx)

        img = Image.open(image_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

