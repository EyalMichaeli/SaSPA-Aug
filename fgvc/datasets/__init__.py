import logging
import copy
import random

from .aircraft_dataset import Planes
from .cub_dataset import CUB
from .car_dataset import Cars
from .dtd_dataset import DTDataset
from .compcars_dataset import CompCars
from .aircraft_biased_dataset import PlanesBiased

from fgvc.util import get_transform
try:
    from cutmix.cutmix import CutMix
except ImportError:
    pass


DATASETS_FILES_PATH = "/mnt/raid/home/eyal_michaeli/git/thesis_utils/fgvc/datasets_files"


def get_datasets(dataset, resize, train_sample_ratio=1.0, aug_json=None, aug_sample_ratio=None, limit_aug_per_image=None, special_aug=None, use_cutmix=False, few_shot=None, print_func=logging.info):
    if special_aug is not None and "-" in special_aug:
        special_aug, cutmix_aug = special_aug.split("-")
        special_aug = special_aug.lower()
        assert cutmix_aug == "cutmix", f"Unsupported cutmix augmentation {cutmix_aug}"
        use_cutmix = True if cutmix_aug.lower() == "cutmix" else False
    
    train_transform = get_transform(resize=resize, phase='train', special_aug=special_aug)
    val_transform = get_transform(resize=resize, phase='val')
    if dataset == 'planes':
        train, val, test = Planes(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image, few_shot=few_shot, print_func=print_func), Planes(split='val', transform=val_transform, print_func=print_func), Planes(split='test', transform=val_transform, print_func=print_func)
    elif dataset == 'cub':
        train, val, test =  CUB(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image, few_shot=few_shot, print_func=print_func), CUB(split='val', transform=val_transform, print_func=print_func), CUB(split='test', transform=val_transform, print_func=print_func)
    elif dataset == 'cars':
        train, val, test =  Cars(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image, few_shot=few_shot, print_func=print_func), Cars(split='val', transform=val_transform, print_func=print_func), Cars(split='test', transform=val_transform, print_func=print_func)
    elif dataset == 'dtd':
        train, val, test =  DTDataset(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image, few_shot=few_shot, print_func=print_func), DTDataset(split='val', transform=val_transform, print_func=print_func), DTDataset(split='test', transform=val_transform, print_func=print_func)
    elif dataset == 'compcars':
        train, val, test =  CompCars(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image, few_shot=few_shot, print_func=print_func), CompCars(split='test', transform=val_transform, print_func=print_func)
    elif dataset == 'compcars-parts':
        train, val, test =  CompCars(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image, dataset_type='parts', few_shot=few_shot, print_func=print_func), CompCars(split='val', transform=val_transform, dataset_type='parts', print_func=print_func), CompCars(split='test', transform=val_transform, dataset_type='parts', print_func=print_func)
    elif dataset == 'planes_biased':
        train, val, test =  PlanesBiased(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image, few_shot=few_shot, print_func=print_func), PlanesBiased(split='val', transform=val_transform), PlanesBiased(split='test', transform=val_transform)
    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))
    
    if use_cutmix or special_aug == "cutmix":
        logging.info("Using CutMix augmentation")
        # we used the same params for cutmix as ALIA, DA-Fusion
        # DA-Fusion: https://github.com/brandontrabucco/da-fusion/blob/main/train_classifier.py#L134
        return CutMix(train, num_class=train.num_classes, beta=1.0, prob=0.5, num_mix=2).dataset, val, test
    else:
        return train, val, test
    
