import sys
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import datetime
import logging
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import clip.clip as clip

try:
    import lpips
except ImportError:
    pass

from fgvc.datasets import *
import all_utils.dataset_utils as dataset_utils


device = torch.device("cuda:0")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, smaller_side_res):
    """
    will resize the image to resolution, while keeping the aspect ratio
    will make the smaller side of the image to be smaller_side_res 
    will keep the size to be a multiple of 64.
    if the output images res is bigger than max_res_size, will resize to 512 smallest side. to avoid feeding SD with too big images
    """
    MAX_RES_SIZE = 1200000  # 1200*1000
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(smaller_side_res) / min(H, W)
    H *= k
    W *= k
    if H * W > MAX_RES_SIZE:
        k = np.sqrt(MAX_RES_SIZE / (H * W))
        H *= k
        W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

apply_canny = CannyDetector()

def preprocess_canny(
    input_image: np.ndarray,
    image_resolution: int,
    low_threshold: int,
    high_threshold: int,
) -> Image.Image:
    image = resize_image(HWC3(input_image), image_resolution)
    control_image = apply_canny(image, low_threshold, high_threshold)
    control_image = HWC3(control_image)
    # vis_control_image = 255 - control_image
    # return PIL.Image.fromarray(control_image), PIL.Image.fromarray(
    #     vis_control_image)
    return Image.fromarray(control_image)


def generate_canny(cond_image_input, low_threshold, high_threshold, image_resolution) -> Image.Image:
    # convert cond_image_input to numpy array
    cond_image_input = np.array(cond_image_input).astype(np.uint8)

    # canny_input, vis_control_image = preprocess_canny(cond_image_input, 512, low_threshold=100, high_threshold=200)
    vis_control_image = preprocess_canny(cond_image_input, image_resolution, low_threshold=low_threshold, high_threshold=high_threshold)

    return vis_control_image 



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, tokenized_prompts):
        x = self.token_embedding(tokenized_prompts).type(self.dtype)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CLIP_selector(nn.Module):
    def __init__(self, clip_model, train_preprocess, val_preprocess, tokenized_prompts):
        super().__init__()
        # self.prompt_learner = PromptLearner(args, classnames, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.train_preprocess = train_preprocess
        self.val_preprocess = val_preprocess
        self.tokenized_prompts = tokenized_prompts

    def forward(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

            # prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(tokenized_prompts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

        return logits


def get_semantic_filtering(image, clip_selector, preprocess, cls_idx=0):
    """
    will return True if CLIP predicts the image to be of the right prompt
    the right class index is always the first
    """
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    logits = clip_selector(image_tensor)
    # make sure the highest confidence is for the right class
    return logits.argmax(dim=-1) == cls_idx


def get_clip_filtering(threshold, image, clip_selector, preprocess, cls_idx):
    # Preprocess the image
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Compute the logits using the clip_selector
    logits = clip_selector(image_tensor)

    # Calculate the confidence (softmax) for the prompt's class
    confidence = torch.softmax(logits, dim=1)[0, cls_idx].item()
        
    # Compare the confidence with the threshold
    return confidence >= threshold


def get_aug_json_path(augmented_image_folder_path, lpips_min=None, lpips_max=None, clip_filtering=False, 
                      clip_filtering_discount=1, semantic_filtering=False, model_confidence_based_filtering=False, 
                      conf_top_k: int = 10, filter_confidence_higher_than: int = None, alia_conf_filtering=False):
    """
    given an augmented image folder path, return the path of the json file with the augmented images paths
    """
    json_name = ""
    if lpips_min:
        json_name += f"lpips_min_{lpips_min}-"
    if lpips_max:
        json_name += f"lpips_max_{lpips_max}-"
    if clip_filtering:
        json_name += f"clip_filtering_{clip_filtering}_discount_{clip_filtering_discount}-"
    if semantic_filtering:
        json_name += f"semantic_filtering-"
    if model_confidence_based_filtering:
        json_name += f"model_confidence_based_filtering_top_{conf_top_k}_classes-"
        if filter_confidence_higher_than:
            json_name += f"filter_confidence_higher_than_{filter_confidence_higher_than}-"
    if alia_conf_filtering:
        json_name += f"alia_conf_filtering-"
    json_name += f"aug.json"

    json_path = str(Path(augmented_image_folder_path).parent / json_name)
    return json_path


def create_json_of_image_name_to_augmented_images_paths(dataset, augmented_image_folder_path, lpips_min=None, lpips_max=None, 
                                                        resize: Tuple = (256, 256), clip_filtering=False, clip_filtering_discount=1,
                                                        semantic_filtering=False, model_confidence_based_filtering=False, 
                                                        conf_top_k: int = 10, filter_confidence_higher_than: int = None, init_log=True, alia_conf_filtering=False):

    assert not (clip_filtering and model_confidence_based_filtering), f"can't use both clip_filtering and model_confidence_based_filtering"

    model_name = f"{Path(augmented_image_folder_path).parent.parent.parent.parent.name}/{Path(augmented_image_folder_path).parent.parent.parent.name}/{Path(augmented_image_folder_path).parent.parent.name}"

    # make sure the folder ends with /images
    if not augmented_image_folder_path.endswith("/images"):
        augmented_image_folder_path = str(Path(augmented_image_folder_path) / "images")
        
    json_path = get_aug_json_path(augmented_image_folder_path, lpips_min, lpips_max, clip_filtering, 
                                  clip_filtering_discount, semantic_filtering, model_confidence_based_filtering, conf_top_k, filter_confidence_higher_than, alia_conf_filtering)

    log_path = json_path.replace(".json", ".log")

    if init_log:
        init_logging(logdir=None, logfile=log_path)
        logging.info(f"log file: {log_path}")

    logging.info(f"\nmodel = {model_name}\n")
    logging.info(f"json_path = {json_path}")

    substrings_to_exclude = ["_source.", "_style.", "_target.", "_control.", "_original.", "_subject.", "subject_"]

    if "sd_xl" in augmented_image_folder_path.lower() or True:
        # some SD XL outputs are corrupted, so we need to check them with PIL (update: can be for other models too)
        check_folder_of_images_with_pil(augmented_image_folder_path, max_delete=50, substrings_to_exclude=substrings_to_exclude)

    if semantic_filtering or clip_filtering:
        model, preprocess = clip.load('RN50', 'cuda', jit=False)

    utils_to_use: dataset_utils.BaseUtils = dataset_utils.DS_UTILS_DICT[dataset](print_func=logging.info)
    # if dataset == "dtd":
    #     utils_to_use.original_images_paths = utils_to_use.all_original_images_paths  # for DTD need to use all images because there are several partitions (in case I'll use other partiotions, not just 1)
    
    original_images_paths_list = utils_to_use.original_images_paths

    if len(list(Path(augmented_image_folder_path).glob("*.*"))) < 10:
        logging.info(f"augmented_image_folder_path = {augmented_image_folder_path} doesn't exist or has less than 10 images")
        logging.info(f"trying a subfolder with /images")
        augmented_image_folder_path = str(Path(augmented_image_folder_path) / "images")
        if len(list(Path(augmented_image_folder_path).glob("*.*"))) < 10:
            raise FileNotFoundError(f"augmented_image_folder_path = {augmented_image_folder_path} doesn't exist or has less than 10 images") 
        logging.info(f"found images in {augmented_image_folder_path}")
        
    if lpips_min or lpips_max:
        lpips_metric = lpips.LPIPS(net='alex').to(device)

    if clip_filtering:
        logging.info(f"using CLIP filtering, of type {clip_filtering}")
        classnames = utils_to_use.get_classes()
        logging.info(f"total number of classes = {len(classnames)}")

        if dataset in ["planes", "planes_biased"]:
            clip_filtering_prompts = ["a photo of a " + name + ", a type of aircraft." for name in classnames]
            image_stem_to_class_dict = utils_to_use.get_image_stem_to_class_str_dict()  # id --> class
        elif dataset == "cars":
            clip_filtering_prompts = ["a photo of a " + name + ", a type of car." for name in classnames]
            image_stem_to_class_dict = utils_to_use.get_image_stem_to_class_str_dict()  # id --> class
        elif dataset == "dtd":
            clip_filtering_prompts = ["a photo of a " + name + ", a type of texture." for name in classnames]
            image_path_to_class_dict = utils_to_use.get_image_path_to_class_str_dict()

        elif dataset == "compcars":
            clip_filtering_prompts = ["a photo of a " + name + ", a type of car." for name in classnames]
            image_path_to_class_dict = utils_to_use.get_image_path_to_class_str_dict()
        elif dataset == "compcars-parts":
            classnames = sorted(list(utils_to_use.part_to_string.values()))
            clip_filtering_prompts = ["a photo of the " + name + ", of a car." for name in classnames]
            image_path_to_class_dict = {image_path: utils_to_use.part_to_string[Path(image_path).parent.name] for image_path in utils_to_use.all_original_images_paths}
        elif dataset == "cub":
            clip_filtering_prompts = ["a photo of a " + name + ", a type of a bird." for name in classnames]
            image_path_to_class_dict = utils_to_use.get_image_path_to_class_str_dict()

        else:
            raise NotImplementedError
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in clip_filtering_prompts]).cuda()
        clip_selector = CLIP_selector(model, preprocess, preprocess, tokenized_prompts)
        threhold = 1 / len(classnames) / clip_filtering_discount
        logging.info(f"using CLIP filtering with threhold = {threhold}")

    if semantic_filtering:
        negative_prompts = ["a photo of an object", "a photo of a scene", "a photo of geometric shapes", "a photo", "an image", "a black photo"]
        logging.info(f"using semantic filtering")
        semantic_filtering_prompts = [utils_to_use.get_basic_prompt()] + negative_prompts
        tokenized_prompts = clip.tokenize(semantic_filtering_prompts).cuda()
        clip_selector_semantic_filtering = CLIP_selector(model, preprocess, preprocess, tokenized_prompts)
        logging.info(f"using semantic filtering with prompts = {semantic_filtering_prompts}")

    if model_confidence_based_filtering:
        logging.info(f"using model_confidence_based_filtering")
        # load baseline model
        baseline_model, val_transform = utils_to_use.load_baseline_model()
        image_path_to_class_id_dict = utils_to_use.get_image_path_to_class_id_dict()
        # conf_top_k = int(utils_to_use.num_classes * conf_top_k)  # it's a percentage of the classes
        conf_top_k = min(conf_top_k, utils_to_use.num_classes)
        logging.info(f"using model_confidence_based_filtering with conf_top_k = {conf_top_k}")
        if filter_confidence_higher_than:
            logging.info(f"using model_confidence_based_filtering with filter_confidence_higher_than = {filter_confidence_higher_than}")

    if alia_conf_filtering:
        logging.info(f"using alia_conf_filtering")
        # load baseline model
        baseline_model, val_transform = utils_to_use.load_baseline_model()
        image_path_to_class_id_dict = utils_to_use.get_image_path_to_class_id_dict()
        class_id_to_conf_threshold: dict = utils_to_use.get_baseline_conf_threshold()

    dict_image_name_to_augmented_images = {}
    total_filtered_lpips = 0
    total_filtered_clip_filtering = 0
    total_filtered_semanitc_filtering = 0
    total_filtered_not_in_top_k_predictions = 0
    total_filtered_too_high_confidence = 0
    total_filtered_alia_wrong_conf_higher_than = 0  # filters images with wrong prediction and confidence higher than the threshold
    total_filtered_alia_correct_conf_higher_than = 0  # filters images with correct prediction and confidence higher than the threshold
    
    lent = len(original_images_paths_list)
    MAX_FILE_NAME_LENGTH = 40  # also in generation code, to not have too long file names. needs to be consistent here too.
    all_file_names = os.listdir(augmented_image_folder_path)
    all_file_names = [file_name for file_name in all_file_names if not any(substring in file_name for substring in substrings_to_exclude)]
    for i, image_path in enumerate(tqdm(original_images_paths_list)):
        if i % 100 == 0:
            logging.info(f"index = {i}/{lent}")
        image_name = Path(image_path).name
        image_name_without_extension = Path(image_name).stem

        augmented_images_paths = []
        for augmented_image_name in all_file_names:
            if image_name_without_extension[:MAX_FILE_NAME_LENGTH] in augmented_image_name:
                augmented_images_paths.append(str(Path(augmented_image_folder_path) / augmented_image_name))

        augmented_images_paths_copy = augmented_images_paths.copy()
        if model_confidence_based_filtering:
            for augmented_image_path in augmented_images_paths_copy:
                correct_label = image_path_to_class_id_dict[image_path]
                image_tensor = val_transform(Image.open(augmented_image_path)).unsqueeze(0).to(device)
                logits: torch.Tensor = baseline_model(image_tensor)[0] # it returns a tuple of results, the first one is the one we want
                # check if the correct label is in the top-k predictions
                if correct_label not in logits.topk(conf_top_k)[1]:
                    augmented_images_paths.remove(augmented_image_path)
                    total_filtered_not_in_top_k_predictions += 1
                    if total_filtered_not_in_top_k_predictions < 10:
                        logging.info(f"Image not in Top {conf_top_k} predictions filtered: \n{augmented_image_path}")
                elif filter_confidence_higher_than:
                    # check if the confidence of the correct label is higher than the threshold
                    correct_label_confidence = torch.softmax(logits, dim=1)[0, correct_label].item()
                    if correct_label_confidence > filter_confidence_higher_than:
                        augmented_images_paths.remove(augmented_image_path)
                        total_filtered_too_high_confidence += 1
                        if total_filtered_too_high_confidence < 10:
                            logging.info(f"Too high confidence = {correct_label_confidence}, filtered: \n{augmented_image_path}")

        if lpips_min or lpips_max:
            lent_before = len(augmented_images_paths)
            augmented_images_paths = [augmented_image_path for augmented_image_path in augmented_images_paths if lpips_min <= calc_lpips_distance(image_path, augmented_image_path, lpips_metric, resize) <= lpips_max]
            lent_after = len(augmented_images_paths)
            total_filtered_lpips += lent_before - lent_after

        if clip_filtering:
            image_id = image_name_without_extension.split("_")[0]
            if dataset in ["planes", "cars", "planes_biased"]:
                class_string = image_stem_to_class_dict[image_id]                
            else:
                class_string = image_path_to_class_dict[image_path]
            class_idx = classnames.index(class_string)

            lent_before = len(augmented_images_paths)
            augmented_images_paths_copy = augmented_images_paths.copy()
            augmented_images_paths = [augmented_image_path for augmented_image_path in augmented_images_paths if get_clip_filtering(threhold, Image.open(augmented_image_path), clip_selector, preprocess, class_idx)]
            lent_after = len(augmented_images_paths)
            total_filtered_clip_filtering += lent_before - lent_after
            if lent_before - lent_after > 0 and total_filtered_clip_filtering < 10:
                clip_filtering_filtered = [path for path in augmented_images_paths_copy if path not in augmented_images_paths]
                clip_filtering_filtered = '\n'.join(clip_filtering_filtered)
                logging.info(f"clip_filtering filtered: \n{clip_filtering_filtered}")

        if semantic_filtering:
            lent_before = len(augmented_images_paths)
            augmented_images_paths_copy = augmented_images_paths.copy()  # to see what was filtered
            augmented_images_paths = [augmented_image_path for augmented_image_path in augmented_images_paths if get_semantic_filtering(Image.open(augmented_image_path), clip_selector_semantic_filtering, preprocess, cls_idx=0)]
            paths_filtered = [path for path in augmented_images_paths_copy if path not in augmented_images_paths]  # check what was filtered
            total_filtered_semanitc_filtering += len(paths_filtered)
            if paths_filtered and total_filtered_semanitc_filtering < 10:
                semantic_filtering_filtered = '\n'.join(paths_filtered)
                logging.info(f"semantic_filtering filtered: \n{semantic_filtering_filtered}")

        if alia_conf_filtering:
            augmented_images_paths_copy = augmented_images_paths.copy()
            for augmented_image_path in augmented_images_paths_copy:
                # filter images with confidence lower than the threshold
                class_id = image_path_to_class_id_dict[image_path]
                class_conf_threshold = class_id_to_conf_threshold[str(class_id)]
                image_tensor = val_transform(Image.open(augmented_image_path)).unsqueeze(0).to(device)
                logits: torch.Tensor = baseline_model(image_tensor)[0] # it returns a tuple of results, the first one is the one we want
                # get max confidence
                max_conf = logits.max().item()
                if max_conf > class_conf_threshold and random.random() > 0.2:  # 20% are not filtered even though should be, according to paper.
                    # now, seperate into correct and wrong predictions for evaluation and logging
                    if logits.argmax() == class_id:
                        total_filtered_alia_correct_conf_higher_than += 1
                        if total_filtered_alia_correct_conf_higher_than < 10:
                            logging.info(f"Correct prediction and confidence higher than the threshold = {max_conf:.1f}, filtered: \n{augmented_image_path}")
                    else:
                        total_filtered_alia_wrong_conf_higher_than += 1
                        if total_filtered_alia_wrong_conf_higher_than < 10:
                            logging.info(f"Wrong prediction and confidence higher than the threshold = {max_conf:.1f}, filtered: \n{augmented_image_path}")
                    if augmented_image_path in augmented_images_paths:
                        augmented_images_paths.remove(augmented_image_path)
                    else:
                        logging.info(f"augmented_image_path = {augmented_image_path} was already removed (?)")
            

        dict_image_name_to_augmented_images[image_name] = augmented_images_paths
    
    # create directory if doesn't exist
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump(dict_image_name_to_augmented_images, f)
    logging.info(f"Finished creating json of image name to augmented images paths in: \n{json_path}")
    
    for filter_name, filters in {
        "lpips_min": [lpips_min, total_filtered_lpips], 
        "lpips_max": [lpips_max, total_filtered_lpips],
        "clip_filtering": [clip_filtering, total_filtered_clip_filtering],
        "semantic_filtering": [semantic_filtering, total_filtered_semanitc_filtering],
        f"not_in_top_{conf_top_k}": [model_confidence_based_filtering, total_filtered_not_in_top_k_predictions],
        "too_high_confidence": [model_confidence_based_filtering, total_filtered_too_high_confidence],
        "alia_correct_conf_higher_than": [alia_conf_filtering, total_filtered_alia_correct_conf_higher_than],
        "alia_wrong_conf_higher_than": [alia_conf_filtering, total_filtered_alia_wrong_conf_higher_than],
        }.items():
        if filters[0]:
            logging.info(f"For filter = {filter_name}, filtered {filters[1]} images")

    logging.info("\n\n")
    dict_num_augmentations_per_image = get_dict_of_value_counts_image_name_to_num_aug_images(dict_image_name_to_augmented_images)
    logging.info(f"dict_num_augmentations_per_image = {dict_num_augmentations_per_image}")
    logging.info("\n\n")
    logging.info(f"json_path = \n{json_path}")

    return json_path


def get_dict_of_value_counts_image_name_to_num_aug_images(dict_image_name_to_augmented_images: dict, load_the_json=False):
    # provide with a value count for num augmentations per image, 
    # so that if there are 100 images, it could be: 20 images: 0 augmentations, 30 images: 1 augmentation, 50 images: 2 augmentations
    if load_the_json:
        with open(dict_image_name_to_augmented_images) as f:
            dict_image_name_to_augmented_images = json.load(f)
    dict_num_augmentations_per_image = {}
    for image_name, augmented_images_paths in dict_image_name_to_augmented_images.items():
        num_augmentations = len(augmented_images_paths)
        if num_augmentations not in dict_num_augmentations_per_image:
            dict_num_augmentations_per_image[num_augmentations] = 1
        else:
            dict_num_augmentations_per_image[num_augmentations] += 1
            
    return dict_num_augmentations_per_image


def merge_aug_jsons(list_of_jsons: list, output_json_path: str):
    """
    given  a list of aug jsons create by create_json_of_image_name_to_augmented_images_paths(), merge the keys (file names) with the same name. 
    i.e, merge the lists of augmented images paths (values of the dicts are a list of augmented images paths)

    """
    if not Path(output_json_path).parent.exists():
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        
    merged_dict = {}
    for json_path in list_of_jsons:
        with open(json_path) as f:
            dict_image_name_to_augmented_images = json.load(f)
        for image_name, augmented_images_paths in dict_image_name_to_augmented_images.items():
            if image_name not in merged_dict:
                merged_dict[image_name] = augmented_images_paths
            else:
                merged_dict[image_name] += augmented_images_paths

    with open(output_json_path, 'w') as f:
        json.dump(merged_dict, f)

    print(f"Finished merging jsons in \n{output_json_path}")

    print(get_dict_of_value_counts_image_name_to_num_aug_images(merged_dict))

    return merged_dict


def delete_files_in_folder_with_substr(folder_path, substr, max_num_files_to_delete=300):
    num_deleted = 0
    for file_name in tqdm(os.listdir(folder_path)):
        if substr in file_name:
            os.remove(f"{folder_path}/{file_name}")
            num_deleted += 1
            if num_deleted >= max_num_files_to_delete:
                print(f"Reched max_num_files_to_delete = {max_num_files_to_delete}, breaking")
                break
                
    print(f"Finished deleting {num_deleted} files in {folder_path} with substr {substr}")


def create_dict_image_path_to_augmented_images_paths(aug_data_folder, original_images_paths):
    dict_image_path_to_augmented_images = {}
    for image_path in tqdm(original_images_paths):
        image_name = Path(image_path).name
        image_name_without_extension = Path(image_name).stem
        augmented_images_paths = [str(Path(aug_data_folder) / augmented_image_name) for augmented_image_name in os.listdir(aug_data_folder) if image_name_without_extension in augmented_image_name and "_source" not in augmented_image_name]
        dict_image_path_to_augmented_images[image_path] = augmented_images_paths
    return dict_image_path_to_augmented_images


def plot(orig_img, imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, figsize=(num_cols * 8, num_rows * 8))
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


def plot_images_in_row(images_list, titles=None):
    """any number of images in a row"""
    plt.figure(figsize=(20, 20))
    columns = len(images_list)
    for i, image in enumerate(images_list):
        plt.subplot(1, columns, i + 1)
        plt.imshow(image)
        if titles is not None:
            plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def calc_lpips_distance(path1, path2, lpips_metric, resize: Tuple = None):
    img1 = Image.open(path1).convert("L").convert("RGB")
    img2 = Image.open(path2).convert("L").convert("RGB")
    # resize images
    if resize:
        img1 = img1.resize(resize)
        img2 = img2.resize(resize)  
    img1 = T.ToTensor()(img1).to(device)
    img2 = T.ToTensor()(img2).to(device)
    img1 = img1*2 - 1  # this is practically the same as img1 = (img1 - 0.5) / 0.5
    img2 = img2*2 - 1
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    dist01 = lpips_metric(img1, img2)
    return dist01


def init_logging(logdir=None, logfile=None, return_logger=False):
    assert logdir or logfile, "logdir or logfile must be provided"
    date_uid = str(datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S"))
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        log_file = os.path.join(logdir, f'{date_uid}_log.log')
    else:
        # append the data at the end of the logfile
        parent_folder = Path(logfile).parent
        parent_folder.mkdir(parents=True, exist_ok=True)
        log_file = str(parent_folder / f"{Path(logfile).stem}_{date_uid}{Path(logfile).suffix}")
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(fh)
    # get the logger 
    if return_logger:
        return logging.getLogger()
    else:
        return logdir


def load_data(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            image_id, info = line.strip().split(' ', 1)
            data[image_id] = info
    return data


def get_same_class_image_names(dataset="planes", num_per_image=1, same_car_direction=False, captions_dict=None, split="train", random_class=False):
    """
    a func that returns a dict of ids, where each id has a list of num_per_image ids from the same class
    :param dataset: "planes" or "cars"
    :param num_per_image: how many images to take from the same class
    :param same_car_direction: if True, will take only images with the same direction (front, back). relevant only for cars dataset
    :param captions_dict: dict of image names to captions. relevant only for cars dataset
    :param split: "train" or "val". relevant only for planes dataset
    :param random_class: if True, will take random classes. 
    :return: dict of ids, where each id has a list of num_per_image ids from the same class
    """
    if dataset == "planes":
        planes = dataset_utils.PlanesUtils(split=split)
        ids, classes = planes.get_ids_and_classes()

    elif dataset == "cars":
        if same_car_direction:
            logging.info("same_car_direction is True")
            assert captions_dict is not None, "if same_car_direction is True, captions must be provided"
        cars = dataset_utils.CarsUtils()
        ids, classes = cars.get_ids_and_classes()
    
    else:
        raise NotImplementedError
        
    df = pd.DataFrame({"id": ids, "clas": classes})

    if dataset == "cars" and same_car_direction:
        captions_df = pd.DataFrame.from_dict(captions_dict, orient="index")
        captions_df.index = captions_df.index.str.split("/").str[-1].str.split(".").str[0]
        captions_df = captions_df.reset_index().rename(columns={"index": "id"})
        df = df.merge(captions_df, on="id", how="inner")

        dict_of_ids = {}
        for id in df.id.values.tolist():
            matching_class_filter = (df.clas == df[df.id == id].clas.values[0]) if not random_class else True
            id_values = df[matching_class_filter & (df["is the back or front of the car shown?"] == df[df.id == id]["is the back or front of the car shown?"].values[0])].id.values
            if len(id_values) < num_per_image:
                logging.info(f"not enough images for id {id}, so taking all {len(id_values)} images")
                dict_of_ids[id] = id_values
            else:  # if there are enough images, take num_per_image random images
                dict_of_ids[id] = random.sample(list(id_values), num_per_image)

        return dict_of_ids

    dict_of_ids = {}

    if not random_class:
        for id in df.id.values.tolist():
            dict_of_ids[id] = random.sample(list(df[df.clas == df[df.id == id].clas.values[0]].id.values), num_per_image)
    else:
        for id in df.id.values.tolist():
            dict_of_ids[id] = random.sample(list(df.id.values), num_per_image)

    return dict_of_ids


def check_folder_of_images_with_pil(folder, max_delete=20, substrings_to_exclude=None):
    """
    checkes with img.verify() that all images in folder are valid, and deletes invalid images
    """
    num_deleted = 0
    file_names = [file_name for file_name in os.listdir(folder) if not any(substring in file_name for substring in substrings_to_exclude)]
    for image_name in tqdm(file_names):
        image_path = Path(folder) / image_name
        try:
            img = Image.open(image_path)
            img.verify()

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Exiting...")
            sys.exit(0)
        except:
            print(f"image {image_path} is corrupted, deleting")
            os.remove(image_path)
            num_deleted += 1
            if num_deleted >= max_delete:
                print(f"reached max_delete = {max_delete}, breaking")
                break
    print(f"Finished checking folder {folder} with PIL, deleted {num_deleted} images")


def merge_aug_jsons_with_amount_per_json(dict_json_amount: dict, output_json_path: str, print_func=print):
    """
    will get a dict of jsons and the amount of images to take from each json, and will merge them
    for example, {json1: 1, json2: 1}
    will take for each image one random image from the value of json 1 and one random image from the value of json 2
    """
    # add 'merged' to the output_json_path
    output_json_path = output_json_path.replace(".json", "-merged.json")
    # make sure each file in the dict is not the same as output path
    assert not any(json_path == output_json_path for json_path in dict_json_amount.keys()), "output_json_path can't be the same as any of the jsons in dict_json_amount"

    if not Path(output_json_path).parent.exists():
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        
    merged_dict = {}
    for json_path, amount in dict_json_amount.items():
        with open(json_path) as f:
            dict_image_name_to_augmented_images = json.load(f)
        
        print_func(f"\nbefore merging, for json: {json_path}\n, dict_image_name_to_augmented_images = {get_dict_of_value_counts_image_name_to_num_aug_images(dict_image_name_to_augmented_images)}\n")
        for image_name, augmented_images_paths in dict_image_name_to_augmented_images.items():
            if image_name not in merged_dict:
                merged_dict[image_name] = random.sample(augmented_images_paths, amount) if amount < len(augmented_images_paths) else augmented_images_paths
            else:
                merged_dict[image_name] += random.sample(augmented_images_paths, amount) if amount < len(augmented_images_paths) else augmented_images_paths

    with open(output_json_path, 'w') as f:
        json.dump(merged_dict, f)

    print_func(f"Finished merging jsons in \n{output_json_path}")

    print_func(get_dict_of_value_counts_image_name_to_num_aug_images(merged_dict))

    return merged_dict


def remove_all_augs_w_sub_str_and_save(json_path: str, substr_to_remove: list, output_json_path: str):
    """
    will remove all images with the substr in the name from the json, and save the new json
    """
    with open(json_path) as f:
        dict_image_name_to_augmented_images = json.load(f)

    print(f"before removing substrings, dict_image_name_to_augmented_images = {get_dict_of_value_counts_image_name_to_num_aug_images(dict_image_name_to_augmented_images)}")

    for image_name, augmented_images_paths in dict_image_name_to_augmented_images.items():
        dict_image_name_to_augmented_images[image_name] = [path for path in augmented_images_paths if not any(substring in path for substring in substr_to_remove)]

    with open(output_json_path, 'w') as f:
        json.dump(dict_image_name_to_augmented_images, f)

    print(f"Finished removing all images with substr in the name from the json, and saving the new json in \n{output_json_path}")

    print(get_dict_of_value_counts_image_name_to_num_aug_images(dict_image_name_to_augmented_images))

    return dict_image_name_to_augmented_images


def calc_lpips_on_2_images(lpips_metric, file_path1, file_path2, resize_to=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    assert Path(file_path1).exists(), f"File not found: {file_path1}"
    assert Path(file_path2).exists(), f"File not found: {file_path2}"
    
    img1 = Image.open(file_path1).convert("RGB")
    img2 = Image.open(file_path2).convert("RGB")
    
    if resize_to:
        img1 = img1.resize(resize_to)
        img2 = img2.resize(resize_to)
    
    img1 = np.array(img1).astype(np.float32) / 255.0
    img2 = np.array(img2).astype(np.float32) / 255.0
    
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        d = lpips_metric(img1, img2, normalize=True)
    
    return torch.sum(d).item()


def calc_lpips_given_aug_json(dataset, aug_json, net, compute_on=3000, resize_to=None):
    try:
        lpips_metric = lpips.LPIPS(net=net).to(device)

        assert Path(aug_json).exists(), f"File not found: {aug_json}"
        with open(aug_json, 'r') as f:
            aug_data = json.load(f)
        
        lpips_values = []
        # each key is a file name, and the value is a list of augmentations (might be empty)
        # sample compute_on number of images
        if len(aug_data) > compute_on:
            aug_data = dict(random.sample(aug_data.items(), compute_on))

        ds_utils: dataset_utils.BaseUtils = dataset_utils.DS_UTILS_DICT[dataset]()
        all_src_paths = ds_utils.original_images_paths
        for file_name, aug_list in tqdm(aug_data.items()):
            for aug in aug_list:
                file_path = [x for x in all_src_paths if file_name in x]
                assert len(file_path) == 1, f"list length: {len(file_path)} for {file_name}"
                lpips_value = calc_lpips_on_2_images(lpips_metric, file_path[0], aug, resize_to=resize_to)
                lpips_values.append(lpips_value)
        
        mean, std = np.mean(lpips_values), np.std(lpips_values)
        print(f"Mean: {mean}, Std: {std}")
        return mean, std, lpips_values
    except:
        print(f"Error in {dataset}, {aug_json}, {net}")
        return None, None, None



if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=1 python /mnt/raid/home/user_name/git/thesis_utils/all_utils/utils.py
    CUDA_VISIBLE_DEVICES=2 nohup python /mnt/raid/home/eyal_michaeli/git/thesis_utils/all_utils/utils.py > /mnt/raid/home/eyal_michaeli/git/thesis_utils/all_utils/utils2.log 2>&1 &
    """
    pass
    # Generate a json file with the augmented images paths
    DATASET = "cars"  
    AUG_FOLDER = "/mnt/raid/home/eyal_michaeli/datasets/dogs/aug_data/controlnet/blip_diffusion/canny/gpt-meta_class_prompt_w_sub_class_style_img_from_diff_img/v1-res_512-num_2-gs_7.5-num_inf_steps_30_controlnet_scale_0.75_low_120_high_200_seed_1/images"

    CLIP_FILTERING_TYPE = None  # out of "per_class" only for now.
    SEMANTIC_FILTERING = 1
    MODEL_CONFIDENCE_BASED_FILTERING = 1
    CONF_TOP_K = 10
    ALIA_CONF_FILTERING = 0
    json_path = create_json_of_image_name_to_augmented_images_paths(
        DATASET,
        augmented_image_folder_path=AUG_FOLDER,
        clip_filtering=CLIP_FILTERING_TYPE,
        semantic_filtering=SEMANTIC_FILTERING,
        model_confidence_based_filtering=MODEL_CONFIDENCE_BASED_FILTERING,
        conf_top_k=CONF_TOP_K,
        alia_conf_filtering=ALIA_CONF_FILTERING
    )
