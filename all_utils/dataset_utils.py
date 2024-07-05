import glob
import json
import warnings
from pathlib import Path
import os
import pandas as pd
import scipy.io as sio
import torchvision.transforms as transforms
import torch
from PIL import Image
# to ignore a pytorch 2 compile logging:
import torch._dynamo
from tqdm import tqdm
torch._dynamo.config.suppress_errors = True

import fgvc.models.cal as cal
from fgvc.datasets import *
from all_utils import utils



device = torch.device("cuda:0")


DATASETS_SUPPORTED = ["planes", "cars", "dtd", "compcars-parts", "cub", "planes_biased"]


class BaseUtils:
    def __init__(self, split="train", root_path: str = "", print_func=print):
        self.name: str = ""
        self.meta_class: str = ""
        self.root_path = Path(root_path)
        self.split = split
        self.print_func = print_func
        self.original_images_paths = []
        self.baseline_model_cp = ""
    
    def get_classes(self):
        "return a list of all classes in the dataset, in string format"
        raise NotImplementedError

    @property
    def num_classes(self):
        """access it with class.num_classes"""
        return len(self.get_classes())
    
    def get_image_path_to_class_str_dict(self):
        """return a dict of image_path --> class_str, to be used for prompts"""
        raise NotImplementedError
    
    def get_image_stem_to_class_str_dict(self):
        """return a dict of image_stem --> class_str, to be used for prompts"""
        raise NotImplementedError
    
    def get_image_path_to_class_id_dict(self, split="train"):
        """
        returns a dict of image_path --> class_id
        used for confidence based filtering
        Important: this should be the same as the dataset used to train the baseline model.
        """
        raise NotImplementedError
    
    def get_basic_prompt(self):
        """used for semantic filtering"""
        raise NotImplementedError
    
    def get_image_path_with_same_class(self, image_path: str):
        """returns a list of image paths that have the same class as the input image"""
        if self.name in ["planes", "cars"]:
            image_path = Path(image_path).stem
            
        class_str = self.image_path_to_class_str_dict[image_path]
        paths_w_same_class = [path for path, cls in self.image_path_to_class_str_dict.items() if cls == class_str]
        if self.name in ["planes", "cars"]:
            paths_w_same_class = [str(self.images_path / f"{path}.jpg") for path in paths_w_same_class]
        return paths_w_same_class

    def get_transform(self, resize=(224, 224)):
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ]) # taken from CAL code. 

    def load_baseline_model(self, resize=(224, 224)):
        # Load ckpt and get state_dict
        checkpoint = torch.load(self.baseline_model_cp)  # only "logs" and "state_dict" keys
        # Load weights
        state_dict = checkpoint['state_dict']
        try:  # can be either resnet50 or resnet101
            baseline_model = cal.WSDAN_CAL(num_classes=len(self.get_classes()))
            if "_orig_mod" in list(state_dict.keys())[-1]:  # this means that torch.compile compiled the model in pytorch 2.x
                baseline_model = torch.compile(baseline_model)  # needed if trained with pyorch 2.x and used torch.compile
            baseline_model.load_state_dict(state_dict)

        except:
            baseline_model = cal.WSDAN_CAL(num_classes=len(self.get_classes()), net="resnet50")
            if "_orig_mod" in list(state_dict.keys())[-1]:  # this means that torch.compile compiled the model in pytorch 2.x
                baseline_model = torch.compile(baseline_model)  # needed if trained with pyorch 2.x and used torch.compile
            baseline_model.load_state_dict(state_dict)
            
        baseline_model = baseline_model.to(device)
        baseline_model.eval()

        transform = self.get_transform(resize=resize)
        return baseline_model, transform
    
    def get_baseline_conf_threshold(self):
        """first checks if the file exists, and returns it if its not None. If it's None, computes it and saves it into alia_confidence_thresholds folder"""
        json_path = Path(f"alia_confidence_thresholds/{self.name}.json")
        if json_path.exists():
            return json.load(open(json_path, "r"))
        else:
            # compute it
            self.print_func(f"Computing the baseline confidence threshold for {self.name} dataset")
            image_path_to_class_id_dict = self.get_image_path_to_class_id_dict()
            # init class_id_to_list_of_confs with empty lists
            class_id_to_list_of_confs = {i: [] for i in range(self.num_classes)}
            baseline_model, transform = self.load_baseline_model()
            for image_path in tqdm(self.original_images_paths):
                image = transform(Image.open(image_path).convert("RGB"))
                with torch.no_grad():
                    output = baseline_model(image.unsqueeze(0).to(device))
                # get conf of correct class
                class_id = image_path_to_class_id_dict[image_path]
                conf = output[0][: , class_id].item()
                class_id_to_list_of_confs[class_id].append(conf)
            # get the mean of each class
            class_id_to_mean_conf = {class_id: sum(conf_list) / len(conf_list) for class_id, conf_list in class_id_to_list_of_confs.items()}
            # print it
            for class_id, mean_conf in class_id_to_mean_conf.items():
                self.print_func(f"Class {class_id} mean conf: {mean_conf}")
            # save it
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            json.dump(class_id_to_mean_conf, open(json_path, "w"))
            self.print_func(f"Saved the mean confidences of the baseline model into {json_path}")
            return class_id_to_mean_conf

    def get_images_for_split_with_no_val(self, split, original_images_paths, dataset_name):
        dataset_name = dataset_name.lower()
        assert dataset_name in ["cars", "compcars_parts", "cub"]
        # if split is val, load the val txt file: 
        assert split in ['train', 'val']
        file_path = os.path.join(str(Path(__file__).parent.parent), 'fgvc', 'datasets_files', f'{dataset_name}_val.txt')
        with open(file_path, 'r') as f:
            val_image_files = [line.strip() for line in f.readlines()]
        
        new_image_files = []
        new_labels = []
        for image_file in original_images_paths:
            if (split == "val" and image_file in val_image_files) or (split == "train" and image_file not in val_image_files):
                new_image_files.append(image_file)
        return new_image_files
    

class PlanesUtils(BaseUtils):
    def __init__(self, split="train", root_path='/mnt/raid/home/eyal_michaeli/datasets/FGVC-Aircraft/fgvc-aircraft-2013b/data', print_func=print):
        super().__init__(split, root_path, print_func=print_func)
        self.name = "planes"
        self.meta_class = "airplane"
        self.images_path = Path(root_path) / "images"
        self.txt_file_path = self.root_path / f'images_{split}.txt'
        self.images_folder = self.root_path / 'images'
        self.manufacturers_file_path = self.root_path / f'images_manufacturer_{split}.txt'
        self.variants_file_path = self.root_path / f'images_variant_{split}.txt'
        with open(self.txt_file_path, "r") as f:
            self.image_names = f.read().splitlines()
        self.original_images_paths = [str(self.images_folder / f"{image_name}.jpg") for image_name in self.image_names]
        self.print_func(f"Loaded {len(self.original_images_paths)} images for planes")
        self.image_path_to_class_str_dict = self.get_image_stem_to_class_str_dict()
        self.baseline_model_cp = "/mnt/raid/home/eyal_michaeli/git/CAL/fgvc/logs/planes/2023_0917_0427_30_base/model_bestacc.pth"

    def get_image_stem_to_class_str_dict(self):
        manufacturers_data = utils.load_data(self.manufacturers_file_path)
        variants_data = utils.load_data(self.variants_file_path)
        
        image_classes_dict = {}
        for image_id in manufacturers_data:
            if image_id in variants_data:
                manufacturer = manufacturers_data[image_id]
                variant = variants_data[image_id]
                manufacturer_variant = f"{manufacturer} {variant}"
                image_classes_dict[image_id] = manufacturer_variant
        
        return image_classes_dict

    def get_image_path_to_class_id_dict(self, split="train"):
        # taken from pytorch dataset directly
        dataset = Planes(split=split, print_func=self.print_func)
        return dict(zip(dataset._image_files, dataset._labels))


    def get_classes(self):
        d = self.image_path_to_class_str_dict
        return list(set(d.values()))

    def get_basic_prompt(self):
        return "a photo of an aircraft"


class CarsUtils(BaseUtils):
    def __init__(self, split="train", root_path='/mnt/raid/home/eyal_michaeli/datasets/stanford_cars', print_func=print):
        super().__init__(split, root_path, print_func=print_func)
        self.name = "cars"
        self.meta_class = "car"
        assert split in ["train", "val", "test"]
        # Convert the root_path to a Path object
        self.images_path = Path(root_path) / "cars_train"

        # Paths for metadata and annotations are inside the 'devkit' directory
        devkit_path = self.root_path / 'devkit'
        self.meta_file_path = devkit_path / 'cars_meta.mat'
        split_to_use = "train" if split == "val" else split  # because there is on val split, we created it from the train split
        self.annots_path = devkit_path / f'cars_{split_to_use}_annos.mat'

        # Path for images
        self.images_folder = self.root_path / f'cars_{split_to_use}'
        self.original_images_paths = glob.glob(f"{self.images_folder}/*.jpg")

        if split in ["train", "val"]:
            self.original_images_paths = self.get_images_for_split_with_no_val(split, self.original_images_paths, self.name)
        self.print_func(f"Loaded {len(self.original_images_paths)} images for cars, split {split}")
        self.image_path_to_class_str_dict = self.get_image_stem_to_class_str_dict()
        self.baseline_model_cp = "/mnt/raid/home/eyal_michaeli/git/CAL/fgvc/logs/cars/2023_0916_1010_12_base/model_bestacc.pth"

    def get_image_stem_to_class_str_dict(self):
        submodel_data = {}
    
        meta_data = sio.loadmat(self.meta_file_path)['class_names']
        for submodel_info in meta_data[0]:
            submodel_name = submodel_info[0]
            submodel_data[len(submodel_data) + 1] = submodel_name

        car_submodel_dict = {}
        
        annotations = sio.loadmat(self.annots_path)['annotations'][0]
        
        for annotation in annotations:
            image_id = Path(annotation[-1][0]).stem
            class_id = annotation[4][0][0]
            
            if class_id in submodel_data:
                submodel_name = submodel_data[class_id]
                car_submodel_dict[image_id] = submodel_name
        
        return car_submodel_dict

    def get_image_path_to_class_id_dict(self, split="train"):
        """taken from pytorch dataset directly"""
        dataset = Cars(split=split, print_func=self.print_func)
        return dict(zip(dataset._image_files, dataset._labels))


    def get_classes(self):
        d = self.get_image_stem_to_class_str_dict()
        return list(set(d.values()))
    
    def get_basic_prompt(self):
        return "a photo of a car"
    


class DTDUtils(BaseUtils):
    def __init__(self, split="train", partition=1, root_path='/mnt/raid/home/eyal_michaeli/datasets/DTD/dtdataset/dtd', print_func=print):
        """
        partition: 1-10
        """
        super().__init__(split, root_path, print_func=print_func)
        self.name = "dtd"
        self.meta_class = "texture"
        self.images_folder = self.root_path / "images"
        self.all_original_images_paths = glob.glob(f"{self.images_folder}/*/*.jpg")

        with open(f"{self.root_path}/labels/{split}{partition}.txt", "r") as f:
            self.original_images_paths = f.read().splitlines()
        self.original_images_paths = [str(self.images_folder / image_name) for image_name in self.original_images_paths]
        self.print_func(f"Loaded {len(self.original_images_paths)} images for DTD split {split} partition {partition}")
        self.print_func(f"Total images in DTD: {len(self.all_original_images_paths)}")
        self.image_path_to_class_str_dict = self.get_image_path_to_class_str_dict()
        self.baseline_model_cp = "/mnt/raid/home/eyal_michaeli/git/thesis_utils/fgvc/logs/dtd/2023_1120_2043_36_base_cutmix_and_classic/model_bestacc.pth"

    def get_classes(self):
        return os.listdir(self.images_folder)
    
    def get_image_path_to_class_str_dict(self):
        return {image_path: Path(image_path).parent.name for image_path in self.all_original_images_paths}

    def get_image_path_to_class_id_dict(self, split="train"):
        dataset = DTDataset(split=split, print_func=self.print_func)
        image_files = [str(file) for file in dataset._image_files]
        dataset_val = DTDataset(split="val", print_func=self.print_func)
        image_files_val = [str(file) for file in dataset_val._image_files]
        dataset_test = DTDataset(split="test", print_func=self.print_func)
        image_files_test = [str(file) for file in dataset_test._image_files]
        image_files.extend(image_files_val)
        image_files.extend(image_files_test)
        labels = dataset._labels + dataset_val._labels + dataset_test._labels
        return dict(zip(image_files, labels))
    
    def get_basic_prompt(self):
        return "a photo of a texture"
    


class CompCarsPartsUtils(BaseUtils):
    def __init__(self, split="train", root_path='/mnt/raid/home/eyal_michaeli/datasets/compcars', print_func=print):
        super().__init__(split, root_path, print_func=print_func)
        self.name = "compcars-parts"
        self.meta_class = "car"
        assert split in ["train", "val", "test"]
        split_to_use = "train" if split == "val" else split  # because there is on val split, we created it from the train split
        self.images_folder = self.root_path / "part"

        dataset_name = "compcars"
        make_model_name_file_path = self.root_path / "misc/make_model_name.mat"

        # get make and model names, and corresponding paths
        mat_file = sio.loadmat(make_model_name_file_path)
        model_names_df = pd.DataFrame(mat_file["model_names"], columns=["model_names"])
        model_names_df["model_names"] = model_names_df["model_names"].apply(lambda x: pd.NA if len(x) == 0 else x.item())
        make_names_df = pd.DataFrame(mat_file["make_names"], columns=["make_names"])
        make_names_df["make_names"] = make_names_df["make_names"].apply(lambda x: pd.NA if len(x) == 0 else x.item())

        all_folders = list(glob.glob(f"{self.images_folder}/*/*"))
        self.full_folder_path_to_make_model = {}
        for folder in all_folders:
            make = make_names_df.iloc[int(folder.split("/")[-2])-1]["make_names"]
            model = model_names_df.iloc[int(folder.split("/")[-1])-1]["model_names"]
            self.full_folder_path_to_make_model[folder] = f"{make} {model}"
            
        # get all paths
        image_ext = ["jpg"]
        image_paths_per_ext = [glob.glob(f"{self.images_folder}/*/*/*/*.{ext}") for ext in image_ext]
        self.all_original_images_paths = [path for paths in image_paths_per_ext for path in paths]

        # get paths for split
        dataset_type_folder = "part"
        split_csv_file = self.root_path / f"train_test_split/{dataset_type_folder}/{split_to_use}.csv"
        all_files_csv_file = self.root_path / f"train_test_split/{dataset_type_folder}/train_and_test.csv"

        # it will have path,label cols
        self.original_images_paths = pd.read_csv(split_csv_file, header=None)[0].values.tolist()
        self.all_original_images_paths = pd.read_csv(all_files_csv_file, header=None)[0].values.tolist()

        if split in ["train", "val"]:
            self.original_images_paths = self.get_images_for_split_with_no_val(split, self.original_images_paths, "compcars_parts")

        self.all_classes = list(set(pd.read_csv(all_files_csv_file, header=None)[1].values.tolist()))
        
        self.all_classes_as_strings = [self.full_folder_path_to_make_model[str(Path(image_path).parent.parent.parent)] for image_path in self.original_images_paths] 
        self.all_classes_as_strings = list(set(self.all_classes_as_strings))
        self.part_to_string = {
            "1": "Headlight",
            "2": "Taillight",
            "3": "Fog light",
            "4": "front"  # in the paper it's Exterior Air intake, but this is more general
        }  # taken from compcars paper
        self.print_func(f"Total number of classes in {dataset_name} dataset: {len(self.all_classes)}")
        self.print_func(f"Loaded {len(self.original_images_paths)} images for {dataset_name} dataset split {split}")
        self.print_func(f"Total images in {dataset_name} dataset: {len(self.all_original_images_paths)}")
        self.image_path_to_class_str_dict = self.get_image_path_to_class_str_dict()
        self.baseline_model_cp = "/mnt/raid/home/eyal_michaeli/git/CAL/fgvc/logs/compcars-parts/2023_1218_0409_03_base-fixed_classes_401/model_bestacc.pth"

    def get_classes(self):
        """returns all classes in the form of a string"""
        return self.all_classes_as_strings
    
    def get_image_path_to_class_str_dict(self):
        return {image_path: self.full_folder_path_to_make_model[str(Path(image_path).parent.parent.parent)] for image_path in self.all_original_images_paths}
    
    def get_image_path_to_class_id_dict(self, split="train"):
        """taken from CAL code"""
        split_csv_file = f"/mnt/raid/home/eyal_michaeli/datasets/compcars/train_test_split/part/{split}.csv"

        self._labels = []
        self._image_files = []
        with open(split_csv_file, 'r') as f:
            for line in f:
                path, label = line.strip().split(',')
                self._image_files.append(str(self.root_path / path))
                self._labels.append(label)

        all_unique_labels_sorted = sorted(list(set(self._labels)))
        self.label_to_class_id_map = {label: i for i, label in enumerate(all_unique_labels_sorted)}
        self._labels = [self.label_to_class_id_map[label] for label in self._labels]

        self.image_path_to_class_id = dict(zip(self._image_files, self._labels))
        return self.image_path_to_class_id

    
    def get_basic_prompt(self, part: str = None):
        if part:
            return f"close up of the {self.part_to_string[str(part)]} of a"
        else:
            return "close up of a car"
    
    def get_image_path_with_same_class(self, image_path: str):
        """re-implemennted only for CompCars (main is in BaseUtils) bc it has 2 types it needs to follow: class and car part"""
        class_str = self.image_path_to_class_str_dict[image_path]
        part = image_path.split("/")[-2]
        # both class_str and part are needed to be the same
        return [path for path, cls in self.image_path_to_class_str_dict.items() if cls == class_str and path.split("/")[-2] == part]



class CUBUtils(BaseUtils):
    def __init__(self, split="train", root_path="/mnt/raid/home/eyal_michaeli/datasets/CUB/CUB_200_2011", print_func=print):
        """taken from CAL code"""
        super().__init__(split, root_path, print_func=print_func)
        self.name = "cub"
        self.meta_class = "bird"
        self.images_folder = self.root_path / "images"
        ds = CUB(self.root_path, split=split)
        self.original_images_paths = ds._image_files
        self.print_func(f"Loaded {len(self.original_images_paths)} images for CUB")
        self.image_path_to_class_str_dict = self.get_image_path_to_class_str_dict()
        self.baseline_model_cp = "/mnt/raid/home/eyal_michaeli/git/CAL/fgvc/logs/cub/2024_0107_1512_40_hparam_search/model_bestacc.pth"

    def get_image_path_to_class_str_dict(self):
        classes_txt = self.root_path / "classes.txt"
        classes_df = pd.read_csv(classes_txt, sep=" ", header=None)
        classes_df.columns = ["class_id", "class_name"]  # note that class_id here starts from 1
        classes_df["class_id"] = classes_df["class_id"].apply(lambda x: x - 1)  # make it start from 0
        classes_df["class_name"] = classes_df["class_name"].apply(lambda x: x.split(".")[1])  # remove the number from the class name
        # create a dict of image_path --> class_str, using original_images_paths and classes_df
        image_path_to_class_str_dict = {}
        for image_path in self.original_images_paths:
            image_class = Path(image_path).parent.name
            class_id = int(image_class.split(".")[0]) - 1  # make it start from 0
            class_str = classes_df.iloc[class_id]["class_name"]
            image_path_to_class_str_dict[image_path] = class_str
        return image_path_to_class_str_dict

    def get_image_path_to_class_id_dict(self, split="train"):
        # taken from pytorch dataset directly
        dataset = CUB(self.root_path, split=split, print_func=self.print_func)
        return dict(zip(dataset._image_files, dataset._labels))

    def get_classes(self):
        d = self.image_path_to_class_str_dict
        all_classes = list(set(d.values()))
        self.print_func(f"Total number of classes in CUB dataset: {len(all_classes)}")
        return all_classes

    def get_basic_prompt(self):
        return "a photo of a bird"


class PlanesBiasedUtils(BaseUtils):
    def __init__(self, split="train", root_path='/mnt/raid/home/eyal_michaeli/datasets/FGVC-Aircraft/fgvc-aircraft-2013b/data', print_func=print):
        super().__init__(split, root_path, print_func=print_func)
        self.name = "planes"
        self.meta_class = "airplane"
        self.txt_file_path = self.root_path / f'images_{split}.txt'
        self.images_folder = self.root_path / 'images'
        
        self.split = split
        csv_file = Path(__file__).parent.parent / "fgvc" / "datasets_files/aircraft_biased_dataset/alia_cotextual_bias_split.csv"
        self.df = pd.read_csv(csv_file)

        self.manufacturers_file_path = self.root_path / f'images_manufacturer_{split}.txt'
        self.variants_file_path = self.root_path / f'images_variant_{split}.txt'

        if self.split in ['train', 'test']:
            self.df = self.df[self.df['Split'] == split] if split != 'extra' else self.df[self.df['Split'] == 'val']
        if self.split == 'val':
            self.df = self.df[self.df['Split'] == split][::2]
        if self.split == 'extra': # remove unbiased examples
            # talk half of val set and move it to train
            self.df = self.df[self.df['Split'] == 'val'][1::2]
            # self.df = pd.concat([self.df[self.df['Split'] == 'train'], extra_df])
        self.image_names = [Path(f).stem for f in self.df['Filename']]

        self.original_images_paths = [str(self.images_folder / f"{image_name}.jpg") for image_name in self.image_names]
        self.print_func(f"Loaded {len(self.original_images_paths)} images for planes biased dataset {split}")
        self.image_path_to_class_str_dict = self.get_image_stem_to_class_str_dict()
        self.baseline_model_cp = "/mnt/raid/home/eyal_michaeli/git/CAL/fgvc/logs/planes/2023_0917_0427_30_base/model_bestacc.pth"
        
    def get_image_stem_to_class_str_dict(self):
        manufacturers_data = utils.load_data(self.manufacturers_file_path)
        variants_data = utils.load_data(self.variants_file_path)
        
        image_classes_dict = {}
        for image_id in manufacturers_data:
            if image_id in variants_data:
                manufacturer = manufacturers_data[image_id]
                variant = variants_data[image_id]
                manufacturer_variant = f"{manufacturer} {variant}"
                image_classes_dict[image_id] = manufacturer_variant
        
        return image_classes_dict

    def get_image_path_to_class_id_dict(self, split="train"):
        # taken from pytorch dataset directly
        dataset = PlanesBiased(split=split, print_func=self.print_func)
        return dict(zip(dataset._image_files, dataset._labels))


    def get_classes(self):
        d = self.image_path_to_class_str_dict
        return list(set(d.values()))

    def get_basic_prompt(self):
        return "a photo of an aircraft"


DS_UTILS_DICT = {
    "planes": PlanesUtils,
    "cars": CarsUtils,
    "dtd": DTDUtils,
    "compcars-parts": CompCarsPartsUtils,
    "cub": CUBUtils,
    "planes_biased": PlanesBiasedUtils
}


if __name__ == "__main__":
    # test
    pass
    # planes = PlanesUtils()
    # cars = CarsUtils(split="val")
    # dtd = DTDUtils()
    # compcars = CompCarsUtils()
    # cub = CUBUtils()
    # planes_biased = PlanesBiasedUtils()
