import logging
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


##############################################
# Center Loss for Attention Regularization
##############################################
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)


##################################
# Metric
##################################
class Metric(object):
    pass


class AverageMeter(Metric):
    def __init__(self, name='loss'):
        self.name = name
        self.reset()

    def reset(self):
        self.scores = 0.
        self.total_num = 0.

    def __call__(self, batch_score, sample_num=1):
        self.scores += batch_score
        self.total_num += sample_num
        return self.scores / self.total_num


class TopKAccuracyMetric(Metric):
    def __init__(self, topk=(1,)):
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.size(0)
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            correct_k = correct[:k].reshape(-1).float().sum(0)  # Use reshape instead of view
            self.corrects[i] += correct_k.item()
        
        res = self.corrects * 100. / self.num_samples
        # make sure the list is at least of length 2, just because the training code assumes it is...
        if len(res) == 1:
            res = np.append(res, 0.0)
        return res


class MeanClassAccuracyMetric:
    def __init__(self, num_classes):
        self.name = 'mean_class_accuracy'
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.corrects = np.zeros(self.num_classes)
        self.counts = np.zeros(self.num_classes)
    
    def __call__(self, output, target):
        """Computes the mean class accuracy"""
        _, pred = output.max(1)  # Get the index of the max log-probability
        for i in range(self.num_classes):  # Loop over each class
            class_mask = (target == i)
            class_corrects = pred[class_mask].eq(target[class_mask]).sum()
            self.corrects[i] += class_corrects.item()
            self.counts[i] += class_mask.sum().item()

        self.counts = np.maximum(self.counts, 1)  # Avoid division by zero
        accuracies = self.corrects / self.counts

        # Handle division by zero in case there are no samples for a class
        accuracies = np.nan_to_num(accuracies)

        return accuracies.mean() * 100.

    def accuracy_per_class(self):
        acc_per_class = self.corrects / self.counts
        acc_per_class = np.nan_to_num(acc_per_class)
        return acc_per_class
    
    def total_accuracy(self):
        return self.corrects.sum() / self.counts.sum() 
    


def get_a_plot_of_num_samples_per_class_vs_class_accuracy(num_samples_per_class: dict, class_accuracies: dict, epoch, output_folder):
    """Plot the number of samples per class vs. class accuracy"""
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    logging.info(f"Plotting the number of samples per class vs. class accuracy, output: {str(output_folder)}")

    # make sure that the keys are sorted (for plotting we give only the values)
    num_samples_per_class = dict(sorted(num_samples_per_class.items()))
    class_accuracies = dict(sorted(class_accuracies.items()))
    num_samples_per_class = list(num_samples_per_class.values())
    class_accuracies = list(class_accuracies.values())

    # plot the results
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Number of samples per class')
    ax1.set_ylabel('Class accuracy', color='tab:blue')
    ax1.scatter(num_samples_per_class, class_accuracies, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('Number of samples per class', color='tab:red')  # we already handled the x-label with ax1
    # ax2.plot(num_samples_per_class, num_samples_per_class, color='tab:red')
    # # no need for the second axis y ticks, turn them off: TODO
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(output_folder + f'/num_samples_per_class_vs_class_accuracy_epoch_{epoch}.png')
    return fig
    

##################################
# Callback
##################################
class Callback(object):
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, *args):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, savepath, monitor='val_topk_accuracy', mode='max'):
        self.savepath = savepath
        self.monitor = monitor
        self.mode = mode
        self.reset()
        super(ModelCheckpoint, self).__init__()

    def reset(self):
        if self.mode == 'max':
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')

    def set_best_score(self, score):
        if isinstance(score, np.ndarray):
            self.best_score = score[0]
        else:
            self.best_score = score

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, logs, net, **kwargs):
        current_score = logs[self.monitor]
        if isinstance(current_score, np.ndarray):
            current_score = current_score[0]

        if (self.mode == 'max' and current_score > self.best_score) or \
            (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score

            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()

            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            if 'feature_center' in kwargs:
                feature_center = kwargs['feature_center']
                feature_center = feature_center.cpu()

                torch.save({
                    'logs': logs,
                    'state_dict': state_dict,
                    'feature_center': feature_center}, self.savepath)
            else:
                torch.save({
                    'logs': logs,
                    'state_dict': state_dict}, self.savepath)


##################################
# augment function
##################################
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear', align_corners=False) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


##################################
# transform in dataset
##################################
def get_transform(resize, phase='train', special_aug=None):
    possible_augs = ["classic", "randaug", "cutmix", "none", "autoaug", "classic_no_color", "no", None, False]
    assert special_aug in possible_augs, f"Unsupported special_aug {special_aug}, possible values: {possible_augs}"
    if phase == 'train':
        if special_aug == 'randaug':
            logging.info('\nIMPORTANT: Using RandAugment\n')
            return transforms.Compose([
                transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
                transforms.RandomCrop(resize),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        
        elif special_aug == 'autoaug':
            logging.info('\nIMPORTANT: Using AutoAugment\n')
            return transforms.Compose([
                transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
                transforms.RandomCrop(resize),
                transforms.AutoAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        elif special_aug == "classic_no_color":
            logging.info('\nIMPORTANT: Using classic augmentation but Not using ColorJitter\n')
            return transforms.Compose([
                transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
                transforms.RandomCrop(resize),
                transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(brightness=0.126, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif special_aug == 'classic':
            logging.info('\nIMPORTANT: Using Default classic Augmentation\n')
            return transforms.Compose([
                transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
                transforms.RandomCrop(resize),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.126, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            logging.info('\nIMPORTANT: Not using ANY augmentation\n')
            return transforms.Compose([
                transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
