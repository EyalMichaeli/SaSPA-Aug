import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

from fgvc.datasets import PlanesBiased  # Assuming PlanesBiased class is properly defined in your_module
from fgvc.models import WSDAN_CAL
from fgvc.util import MeanClassAccuracyMetric, TopKAccuracyMetric, get_transform


def load_model(model_path, num_classes):
    net = WSDAN_CAL(num_classes=num_classes, M=32, net="resnet50", pretrained=False)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    if "_orig_mod" in list(state_dict.keys())[-1]:  # this means that torch.compile compiled the model in pytorch 2.x
        net = torch.compile(net)  # needed if trained with pyorch 2.x and used torch.compile
    net.load_state_dict(state_dict)
    net.cuda()
    net.eval()
    return net


def validate_model(model, data_loader, num_classes):
    mean_class_acc_metric = MeanClassAccuracyMetric(num_classes)
    regular_acc_metric = TopKAccuracyMetric()
    id_acc_metric = TopKAccuracyMetric()
    ood_metric = TopKAccuracyMetric()
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.cuda(), labels.cuda()
            y_pred_raw, y_pred_aux, feature_matrix, attention_map = model(images)
            # Assume 'labels' include domain information, 0 for in-domain, 1 for OOD
            in_domain_idx = labels[:, 1] == 0
            ood_idx = labels[:, 1] == 1
            mean_class_acc = mean_class_acc_metric(y_pred_raw, labels[:, 0])  # Update mean class accuracy
            regular_acc = regular_acc_metric(y_pred_raw, labels[:, 0])  # Update regular accuracy
            id_acc = id_acc_metric(y_pred_raw[in_domain_idx], labels[in_domain_idx][:, 0])  # Update in-domain accuracy
            ood_acc = ood_metric(y_pred_raw[ood_idx], labels[ood_idx][:, 0])  # Update OOD accuracy
    print(f"Total samples in regular: {regular_acc_metric.num_samples}")
    print(f"Total samples in in-domain: {id_acc_metric.num_samples}")
    print(f"Total samples in OOD: {ood_metric.num_samples}")
    return mean_class_acc, regular_acc, id_acc, ood_acc


def main(model_path, batch_size=128):
    transform = get_transform(resize=(224, 224), phase='val')
    dataset = PlanesBiased(split='test', transform=transform)
    # make the labels in the dataset contain domain information. 0 for in-domain, 1 for OOD
    # in the dataset.df, add a column 'is_ood' with 0 for in-domain and 1 for OOD
    # for the df, if Label == 1 and Group == 2 or label == 0 and Group == 1, then it's OOD (is_ood = 1)
    dataset.df['is_ood'] = 0
    dataset.df.loc[(dataset.df.Plane == "Boeing") & (dataset.df.Ground.isin(["road"])), 'is_ood'] = 1
    dataset.df.loc[(dataset.df.Plane == "Airbus") & (dataset.df.Ground.isin(["grass"])), 'is_ood'] = 1
    # Now, the dataset.df has a column 'is_ood' with 0 for in-domain and 1 for OOD
    # create new _image_files and _labels based on the new 'is_ood' column
    dataset._image_files = np.array([os.path.join(dataset.images_path, Path(f).name) for f in dataset.df['Filename']])
    dataset._labels = np.array(dataset.df['Label'])
    dataset._is_ood = np.array(dataset.df['is_ood'])
    # combine the labels and is_ood
    dataset._labels = np.stack([dataset._labels, dataset._is_ood], axis=1)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    try:
        model = load_model(model_path, dataset.num_classes)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    mean_class_acc, regular_acc, in_domain_acc, ood_acc = validate_model(model, data_loader, dataset.num_classes)
    print(f"Results for {Path(model_path).parent.name}")
    print(f"Mean Class Accuracy: {mean_class_acc}")
    print(f"Regular Accuracy: {regular_acc}")
    print(f"In-domain Accuracy: {in_domain_acc}")
    print(f"Out-of-domain Accuracy: {ood_acc}")


if __name__ == "__main__":

    cp_folder = "/mnt/raid/home/eyal_michaeli/git/thesis_utils/fgvc/logs/planes_biased"
    # run it for every checkpoit in the folder. the folder has subfolders, and some of them has checkpoints (ends with .pth)
    # if a sub folder has a checkpoint,, run on it and print the results, together with the sub folder name
    for folder in tqdm(Path(cp_folder).iterdir()):
        if folder.is_dir():
            cp_files = list(folder.glob("*.pth"))
            if len(cp_files) > 0:
                print(f"Running on {folder}")
                main(str(cp_files[0]))
                print("Finished running")
                print("=====================================")
    print("Done")
