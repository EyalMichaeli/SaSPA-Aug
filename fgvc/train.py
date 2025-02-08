import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# also for user warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import datetime
import os
import sys
from pathlib import Path
import traceback
import numpy as np
import time
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import argparse
import wandb
from torch.cuda.amp import GradScaler
from torch import autocast

torch.tensor([1.0]).cuda()  # this is to initialize cuda, so that the next cuda calls will not be slow. This can also prevent bugs

# to ignore a pytorch 2 compile logging:
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

sys.path.append(str(Path(__file__).parent.parent))

from fgvc.models import WSDAN_CAL
from fgvc.util import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment, MeanClassAccuracyMetric, get_a_plot_of_num_samples_per_class_vs_class_accuracy
from fgvc.datasets import get_datasets
from fgvc.configs import base_args


def seed_worker(worker_id):
    # worker_id is important as it is used to set the seed for the specific worker in the dataloader
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--dataset', type=str, default='planes')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--weight_decay', type=float, default=None, help="default 1e-5 (hardcoded in the code!), if u want to change it, change it in the code")
    parser.add_argument('--net', type=str, default="resnet101")
    # augmentation options
    parser.add_argument("--aug_json", type=str, default=None,
                        help="path to augmentation json file")
    parser.add_argument("--aug_sample_ratio", type=float, default=None,
                        help="ratio to augment the original image")
    parser.add_argument("--limit_aug_per_image", type=int, default=None,
                        help="limit augmentations per image, default None, which is take all")
    parser.add_argument("--stop_aug_after_epoch", type=int, default=None,
                        help="stop augmenting after this epoch")
    parser.add_argument("--special_aug", type=str, default="classic",
                        help="traditional aug")
    # add arg to take only some amount for the train set
    parser.add_argument("--train_sample_ratio", type=float, default=1.0,
                        help="ratio of train set to take")
    parser.add_argument("--dont_use_wsdan", action="store_true", default=False,
                        help="Don't use wsdan augmentation")
    parser.add_argument("--use_cutmix", action="store_true", default=False,
                        help="Use cutmix augmentation")
    parser.add_argument("--use_target_soft_cross_entropy", action="store_true", default=False,
                        help="Use soft target cross entropy loss")
    parser.add_argument("--few_shot",type=int, default=None,
                        help="K in few-shot learning")
    args = parser.parse_args()
    return args


def import_dataset_config(args):
    if args.dataset in ['planes', 'planes_biased']:
        import fgvc.configs.config_planes as config
    elif args.dataset == 'cars':
        import fgvc.configs.config_cars as config
    elif args.dataset == 'dtd':
        import fgvc.configs.config_dtd as config
    elif args.dataset == 'compcars':
        import fgvc.configs.config_compcars as config
    elif args.dataset == 'compcars-parts':
        import fgvc.configs.config_compcars_parts as config
    elif args.dataset == 'cub':
        import fgvc.configs.config_cub as config
    else:
        raise ValueError('Unsupported dataset {}'.format(args.dataset))
    return config

    
# General loss functions
cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()

# loss and metric
loss_container = AverageMeter(name='loss')
top1_container = AverageMeter(name='top1')
top5_container = AverageMeter(name='top5')

raw_metric = TopKAccuracyMetric()
crop_metric = TopKAccuracyMetric()
drop_metric = TopKAccuracyMetric()

best_val_acc = 0.0
best_test_acc = 0.0


def init_logging(logdir):
    r"""
    Create log directory for storing checkpoints and output images.
    Given a log dir like logs/test_run, creates a new directory logs/2020_0101_1234_test_run

    Args:
        logdir (str): Log directory name
    """
    # log dir
    date_uid = str(datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S"))
    # Remove existing handlers (if any)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logdir_path = Path(logdir)
    logdir = str(logdir_path.parent / f"{date_uid}_{logdir_path.name}")
    os.makedirs(logdir, exist_ok=True)
    # log file
    log_file = os.path.join(logdir, 'log.log')
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(fh)
    logging.info(f"Logging to {log_file}")
    return logdir


def main(args):

    ##################################
    # Logging setting
    ##################################
    try:
        global config
        config = import_dataset_config(args)
        config.save_dir = init_logging(args.logdir)

        # only if stated in args:
        config.epochs = args.epochs if args.epochs else config.epochs
        config.learning_rate = args.learning_rate if args.learning_rate else config.learning_rate
        config.batch_size = args.batch_size if args.batch_size else config.batch_size
        config.weight_decay = args.weight_decay if args.weight_decay else config.weight_decay
        config.net = args.net if args.net else config.net

        if not DONT_WANDB:
            wandb.init(project=f"CAL-aug-exp-new_datasets", name=Path(config.save_dir).name)

        args.net = config.net
        args.image_size = config.image_size
        args.num_attentions = config.num_attentions
        args.beta = config.beta
        args.run_name = '\n'.join(Path(args.logdir).name.split('-'))
        args.logdir = Path(args.logdir).parent.name + "/" + Path(args.logdir).name
        if not args.learning_rate:
            args.learning_rate = config.learning_rate
        if not args.batch_size:
            args.batch_size = config.batch_size
        if not args.weight_decay:
            args.weight_decay = config.weight_decay

        logging.info(f"args: \n{args.__dict__} \n")
        # log args to wandb
        if not DONT_WANDB:
            wandb.config.update(args)

        if DEBUG:
            try:
                import lovely_tensors as lt
                lt.monkey_patch()
            except Exception as e:
                pass

        if args.few_shot in [None, False, 0]:
            args.few_shot = None
        else:
            config.epochs = 100
            # config.batch_size = 32
            # config.learning_rate = 0.001
            # config.weight_decay = 1e-5
            logging.info(f"Few shot learning with K={args.few_shot}")

        # set gpu id
        logging.info(f"gpu_id: {args.gpu_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

        # prints 
        logging.info("PID: {}".format(os.getpid()))
        logging.info(f"Using AMP: {USE_AMP}")

        if args.seed:
            # Setup random seed
            logging.info("Using seed: {}".format(args.seed))
            torch.manual_seed(args.seed)
            random.seed(args.seed)   
            np.random.seed(args.seed)
            torch.cuda.seed_all()
            os.environ['PYTHONHASHSEED'] = str(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            g = torch.Generator()  # used in dataloader
            g.manual_seed(args.seed)  

        if args.dont_use_wsdan:
            logging.info("Not using wsdan augmentation")

        train_dataset, validate_dataset, test_dataset = get_datasets(args.dataset, config.image_size, train_sample_ratio=args.train_sample_ratio, 
                                                                aug_json=args.aug_json, aug_sample_ratio=args.aug_sample_ratio, limit_aug_per_image=args.limit_aug_per_image,
                                                                special_aug=args.special_aug, use_cutmix=args.use_cutmix, few_shot=args.few_shot, print_func=logging.info)

        num_classes = train_dataset.num_classes
        global mean_class_acc
        mean_class_acc = MeanClassAccuracyMetric(num_classes) if any(substr in args.dataset for substr in ["compcars", "bias"]) else None  # "bias is for biased planes dataset"
        if mean_class_acc:
            logging.info(f"Using mean class accuracy metric")

        ##################################
        # Initialize model
        ##################################
        logs = {}
        start_epoch = 0
        net = WSDAN_CAL(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True)
        if not DEBUG:
            # only if pytorch 2.0 is installed
            if int(torch.__version__[0]) >= 2:
                try:
                    logging.info(f"Using torch compile, pytorch version: {torch.__version__}")
                    logging.info(f"You have torch > 2.0.0 you can use torch.compile. if so, uncomment the beow line, pytorch version: {torch.__version__}. For now, not compiling.")
                    # net = torch.compile(net)  # only if pytorch 2.0 is installed
                    logging.info("Done compiling the model")
                except Exception as e:
                    logging.info(f"Failed to compile the model, error: {e}")
                    logging.info(traceback.format_exc())
                    logging.info("Continuing without torch compile")
            else:
                logging.info(f"Pytorch 2.0 is not installed, not using torch compile, pytorch version: {torch.__version__}")

        # init model for soft target cross entropy
        if args.use_target_soft_cross_entropy:
            from all_utils.dataset_utils import PlanesUtils, CarsUtils
            import clip
            import losses
            # the import here because in utils.py there is an initialization of a device, which interrupt the cuda device setting in the start of this script

            logging.info("IMPORTANT: Using soft target cross entropy loss")
            global soft_target_cross_entropy, clip_selector, image_stem_to_class_dict

            soft_target_cross_entropy = losses.SoftTargetCrossEntropy_T()  # input is logits, CLIP logits
            model, preprocess = clip.load('RN50', 'cuda', jit=False)
            if args.dataset == "planes":
                planes = PlanesUtils(print_func=logging.info)
                classnames = planes.get_classes()
                prompts = ["a photo of a " + name + ", a type of aircraft." for name in classnames]
                image_stem_to_class_dict = planes.get_image_stem_to_class_dict()  # id --> class
            elif args.dataset == "cars":
                cars = CarsUtils(print_func=logging.info)
                classnames = cars.get_classes()
                prompts = ["a photo of a " + name + ", a type of car." for name in classnames]
                image_stem_to_class_dict = cars.get_image_stem_to_class_dict()  # id --> class

            
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to("cuda")
            clip_selector = losses.CLIP_selector(model, preprocess, preprocess, tokenized_prompts)

        # feature_center: size of (#classes, #attention_maps * #channel_features)
        feature_center = torch.zeros(num_classes, config.num_attentions * net.num_features).to("cuda")
        

        if config.ckpt and os.path.isfile(config.ckpt):
            # Load ckpt and get state_dict
            checkpoint = torch.load(config.ckpt, weights_only=False)

            # Get epoch and some logs
            logs = checkpoint['logs']
            start_epoch = int(logs['epoch']) # start from the beginning

            # Load weights
            state_dict = checkpoint['state_dict']
            net.load_state_dict(state_dict)
            logging.info('Network loaded from {}'.format(config.ckpt))
            logging.info('Network loaded from {} @ {} epoch'.format(config.ckpt, start_epoch))

            # load feature center
            if 'feature_center' in checkpoint:
                feature_center = checkpoint['feature_center'].to("cuda")
                logging.info('feature_center loaded from {}'.format(config.ckpt))

        logging.info('Network weights save to {}'.format(config.save_dir))

        net.to("cuda")

        learning_rate = config.learning_rate
        logging.info(f"Learning rate: {learning_rate}")
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)


        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                                num_workers=config.workers, pin_memory=True, drop_last=True, shuffle=True, worker_init_fn=seed_worker, generator=g)
        validate_loader = DataLoader(validate_dataset, batch_size=config.batch_size * 2,
                                                num_workers=config.workers, pin_memory=True, drop_last=True, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 2, 
                                 num_workers=config.workers, pin_memory=True, drop_last=True, shuffle=False) if test_dataset else None

        callback_monitor = 'val_{}'.format(raw_metric.name)
        callback = ModelCheckpoint(savepath=os.path.join(config.save_dir, config.model_name),
                                    monitor=callback_monitor,
                                    mode='max')
        if callback_monitor in logs:
            callback.set_best_score(logs[callback_monitor])
        else:
            callback.reset()
            logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Val size: {}'.
                        format(config.epochs, config.batch_size, len(train_dataset), len(validate_dataset)))
            logging.info('')

        logging.info("PID: {}".format(os.getpid()))
        logging.info("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

        scaler = GradScaler(enabled=USE_AMP)
        best_val_acc_list = []
        for epoch in tqdm(range(start_epoch, config.epochs)):
            if args.aug_json and args.stop_aug_after_epoch and epoch >= args.stop_aug_after_epoch:
                train_dataset.stop_aug = True
                logging.info(f"Reached args.stop_aug_after_epoch={args.stop_aug_after_epoch}, stopped augmentation")
                
            logging.info("\n")
            callback.on_epoch_begin()
            logs['epoch'] = epoch + 1
            logs['lr'] = optimizer.param_groups[0]['lr']

            logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

            pbar = tqdm(total=len(train_loader), unit=' batches')
            pbar.set_description('Epoch {}/{}'.format(epoch + 1, config.epochs))

            train(epoch=epoch,
                logs=logs,
                data_loader=train_loader,
                net=net,
                feature_center=feature_center,
                optimizer=optimizer,
                pbar=pbar,
                args=args,
                scaler=scaler)
            # plot = get_a_plot_of_num_samples_per_class_vs_class_accuracy(train_dataset, net, "cuda", output_folder=f"{config.save_dir}/plots/train", epoch=epoch)
            # wandb.log({"train_num_samples_per_class_vs_class_accuracy": wandb.Image(plot)})

            if ((epoch) % 10 == 0 or epoch >= config.epochs - 1 or epoch == config.epochs - 5):  # every 10 epochs or last 3 epochs
                best_val_acc = validate(logs=logs,
                        data_loader=validate_loader,
                        net=net,
                        pbar=pbar,
                        epoch=epoch,
                        is_test=False, 
                        output_folder=f"{config.save_dir}/plots/val",
                        args=args)
                best_val_acc_list.append(best_val_acc)
                # plot = get_a_plot_of_num_samples_per_class_vs_class_accuracy(validate_dataset, net, "cuda", output_folder=f"{config.save_dir}/plots/val", epoch=epoch)
                # wandb.log({"val_num_samples_per_class_vs_class_accuracy": wandb.Image(plot)})

                if test_loader:
                    validate(logs=logs,
                            data_loader=test_loader,
                            net=net,
                            pbar=pbar,
                            epoch=epoch,
                            is_test=True,
                            output_folder=f"{config.save_dir}/plots/test",
                            args=args)
                    # plot = get_a_plot_of_num_samples_per_class_vs_class_accuracy(test_dataset, net, "cuda", output_folder=f"{config.save_dir}/plots/test", epoch=epoch)
                    # wandb.log({"test_num_samples_per_class_vs_class_accuracy": wandb.Image(plot)})

            callback.on_epoch_end(logs, net, feature_center=feature_center)
            pbar.close()

            # if best val acc has not improved in the last 20 epochs, stop training
            if len(best_val_acc_list) > 20 and best_val_acc_list[-1] < best_val_acc_list[-20]:
                logging.info("Validation accuracy has not improved in the last 20 epochs, stopping training")
                break
    
    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt at epoch {}'.format(epoch + 1))

    except Exception as e:
        logging.info(traceback.format_exc())
        raise


def adjust_learning(optimizer, epoch, iter):
    """Decay the learning rate based on schedule"""
    base_lr = config.learning_rate
    base_rate = 0.9
    base_duration = 2.0
    lr = base_lr * pow(base_rate, (epoch + iter) / base_duration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(**kwargs):
    # Retrieve training configuration
    epoch = kwargs['epoch']
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    feature_center = kwargs['feature_center']
    optimizer = kwargs['optimizer']
    pbar = kwargs['pbar']
    args = kwargs['args']
    scaler: GradScaler = kwargs['scaler']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    crop_metric.reset()
    drop_metric.reset()
    if mean_class_acc:
        mean_class_acc.reset()

    # begin training
    start_time = time.time()
    net.train()
    batch_len = len(data_loader)
    for i, (X, y) in tqdm(enumerate(data_loader), total=len(data_loader), unit=' batches', desc='Epoch {}/{}'.format(epoch + 1, config.epochs)):
        if DEBUG and i > 50:
            break

        float_iter = float(i) / batch_len
        adjust_learning(optimizer, epoch, float_iter)
        now_lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        # obtain data for training
        X = X.to("cuda")
        y = y.to("cuda")

        y_pred_raw, y_pred_aux, feature_matrix, attention_map = net(X)

        # Update Feature Center
        feature_center_batch = F.normalize(feature_center[y], dim=-1)
        feature_center[y] += config.beta * (feature_matrix.detach() - feature_center_batch)

        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
            drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
        aug_images = torch.cat([crop_images, drop_images], dim=0)
        y_aug = torch.cat([y, y], dim=0)

        with autocast(dtype=torch.float16, device_type="cuda", enabled=USE_AMP):
            # crop images forward
            y_pred_aug, y_pred_aux_aug, _, _ = net(aug_images)

            y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)
            y_aux = torch.cat([y, y_aug], dim=0)

            # loss
            if not args.dont_use_wsdan:  # use wsdan augmentation loss
                REGULAR_CE_RATIO = 0.5
                if args.use_target_soft_cross_entropy:
                    batch_loss = center_loss(feature_matrix, feature_center_batch)  # not realated to CE loss 
                    batch_loss += REGULAR_CE_RATIO * ( cross_entropy_loss(y_pred_raw, y) / 3. + \
                                cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. + \
                                cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. )  # regular CE loss
                    
                    # add soft target cross entropy loss
                    global soft_target_cross_entropy, clip_selector
                    logits = clip_selector(X)

                    logits_aug = torch.cat([logits, logits], dim=0)  # same as y_aug
                    logits_aux = torch.cat([logits, logits_aug], dim=0)  # same as y_aux
                    batch_loss += (1 - REGULAR_CE_RATIO) * ( soft_target_cross_entropy(y_pred_raw, logits) / 3. + \
                                soft_target_cross_entropy(y_pred_aux, logits_aux) * 3. / 3. + \
                                soft_target_cross_entropy(y_pred_aug, logits_aug) * 2. / 3. ) # soft target CE loss

                else: # regular loss with normal CE
                    batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
                                cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. + \
                                cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. + \
                                center_loss(feature_matrix, feature_center_batch)
            else:
                # not divinding by 3 because not using 3 diff losses. This is not efficient because still computing it, just no using it.
                batch_loss = cross_entropy_loss(y_pred_raw, y) + center_loss(feature_matrix, feature_center_batch)

            pass # end of loss calculation and forward (with autocast)

        # backward
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred_raw, y)
            epoch_crop_acc = crop_metric(y_pred_aug, y_aug)
            epoch_drop_acc = drop_metric(y_pred_aux, y_aux)
            if mean_class_acc:
                epoch_mean_class_acc = mean_class_acc(y_pred_raw, y)

    # end of this epoch
    last_batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Aug Acc ({:.2f}, {:.2f}), Aux Acc ({:.2f}, {:.2f}), lr {:.5f}'.format(
        epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1],
        epoch_crop_acc[0], epoch_crop_acc[1], epoch_drop_acc[0], epoch_drop_acc[1], now_lr)
    if mean_class_acc:
        last_batch_info += ', Mean Class Acc {:.2f}'.format(epoch_mean_class_acc)

    pbar.update()
    pbar.set_postfix_str(last_batch_info)

    # end of this epoch
    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
    logs['train_crop_{}'.format(crop_metric.name)] = epoch_crop_acc
    logs['train_drop_{}'.format(drop_metric.name)] = epoch_drop_acc
    if mean_class_acc:
        logs['train_mean_class_acc'] = epoch_mean_class_acc
    logs['train_info'] = last_batch_info
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if not DONT_WANDB:
        # wandb
        dict_to_log = {
            'train_loss': epoch_loss,
            'train_raw_acc': epoch_raw_acc[0],
            'train_crop_acc': epoch_crop_acc[0],
            'train_drop_acc': epoch_drop_acc[0],
            'train_lr': now_lr,
            'epoch': epoch,
            'epoch_time': total_time
        }
        if mean_class_acc:
            dict_to_log['train_mean_class_acc'] = epoch_mean_class_acc
        wandb.log(dict_to_log)

    # write log for this epoch
    logging.info('Train: {}'.format(last_batch_info))
    # time
    logging.info('Total epoch Time: {}'.format(total_time_str))
    


def validate(**kwargs):
    # Retrieve training configuration
    global best_val_acc
    global best_test_acc
    epoch = kwargs['epoch']
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']
    is_test = kwargs['is_test']
    output_folder = kwargs['output_folder']
    args = kwargs['args']

    val_str = 'test' if is_test else 'val'

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    drop_metric.reset()
    if mean_class_acc:
        mean_class_acc.reset()

    # begin validation
    start_time = time.time()
    net.eval()
    logging.info('Start validating')
    class_to_num_samples_num_corrects_dict = {}  # will be the length of the num of classes. will be {class: [num_samples, num_corrects]}
    with torch.no_grad():
        for i, (X, y) in tqdm(enumerate(data_loader)):
            if DEBUG and i > 100:
                break
            # obtain data
            X = X.to("cuda")
            y = y.to("cuda")

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, y_pred_aux, _, attention_map = net(X)
            
            # update class_to_num_samples_num_corrects
            preds = torch.argmax(y_pred_raw, dim=1)
            for label, pred in zip(y, preds):
                label_item = label.item()
                pred_item = pred.item()
                if label_item not in class_to_num_samples_num_corrects_dict:
                    class_to_num_samples_num_corrects_dict[label_item] = [0, 0]
                class_to_num_samples_num_corrects_dict[label_item][0] += 1
                class_to_num_samples_num_corrects_dict[label_item][1] += (pred_item == label_item)
            
            crop_images3 = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop3, y_pred_aux_crop3, _, _ = net(crop_images3)

            ##################################
            # Final prediction
            ##################################
            y_pred = (y_pred_raw + y_pred_crop3) / 2.
            y_pred_aux = (y_pred_aux + y_pred_aux_crop3) / 2.

            # loss
            batch_loss = cross_entropy_loss(y_pred, y)
            batch_loss = batch_loss.data
            epoch_loss = loss_container(batch_loss.item())

            y_pred = y_pred
            y_pred_aux = y_pred_aux
            y = y

            # metrics: top-1,5 error
            epoch_acc = raw_metric(y_pred, y)
            aux_acc = drop_metric(y_pred_aux, y)
            if mean_class_acc:
                epoch_mean_class_acc = mean_class_acc(y_pred, y)

    # end of validation
    logs[f'{val_str}_{loss_container.name}'] = epoch_loss
    logs[f'{val_str}_{raw_metric.name}'] = epoch_acc
    if mean_class_acc:
        logs[f'{val_str}_mean_class_acc'] = epoch_mean_class_acc

    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    batch_info = f'{val_str} Loss {epoch_loss:.4f}, {val_str} Acc ({epoch_acc[0]:.2f}, {epoch_acc[1]:.2f})'

    pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))

    current_split_best_acc = best_test_acc if is_test else best_val_acc
    if not is_test:
        if epoch_acc[0] > best_val_acc:  # save best model, only in validation
            current_split_best_acc = epoch_acc[0]
            best_val_acc = epoch_acc[0]
            save_model(net, logs, 'model_bestacc.pth')

    if is_test:
        if epoch_acc[0] > best_test_acc:
            current_split_best_acc = epoch_acc[0]
            best_test_acc = epoch_acc[0]

    if epoch % 30 == 0:
        # TODO: currently the samples per class is WRT vla set. doesnt make sense, should use samples per class of train set.
        # so, don't use it till you fix it
        # get the acc per class
        pass
        # class_to_acc = {}
        # for class_id, (num_samples, num_corrects) in class_to_num_samples_num_corrects_dict.items():
        #     class_to_acc[class_id] = num_corrects / max(num_samples, 1)  # to avoid division by zero
        # class_to_num_samples = {class_id: num_samples for class_id, (num_samples, _) in class_to_num_samples_num_corrects_dict.items()}
        # # plot the acc per class
        # plot = get_a_plot_of_num_samples_per_class_vs_class_accuracy(class_to_num_samples, class_to_acc, epoch, output_folder)
        # if not DONT_WANDB:
        #     wandb.log({f"{val_str}_num_samples_per_class_vs_class_accuracy_epoch_{epoch}": wandb.Image(plot)})

    if not DONT_WANDB:
        # wandb
        dict_to_log = {
            f'{val_str}_loss': epoch_loss,
            f'{val_str}_raw_acc': epoch_acc[0],
            f'{val_str}_best_raw_acc': current_split_best_acc,
            f'{val_str}_crop_acc': aux_acc[0],
            f'{val_str}_drop_acc': aux_acc[0],
            'epoch': epoch,
            f'{val_str}_time': total_time
        }
        if mean_class_acc:
            dict_to_log[f'{val_str}_mean_class_acc'] = epoch_mean_class_acc
        
        wandb.log(dict_to_log)

    # if epoch % 10 == 0:
    #     save_model(net, logs, 'model_epoch%d.pth' % epoch)

    if epoch > 30 and best_val_acc < 2:
        logging.info("Validation accuracy is too low, stopping training")
        exit()

    batch_info = f'{val_str} Loss {epoch_loss:.4f}, {val_str} Acc ({epoch_acc[0]:.2f}, {epoch_acc[1]:.2f}), {val_str} Aux Acc ({aux_acc[0]:.2f}, {aux_acc[1]:.2f}), Best {current_split_best_acc:.2f}'
    logging.info(batch_info)

    # write log for this epoch
    logging.info('Valid: {}'.format(batch_info))
    logging.info('Total Val Time: {}'.format(total_time_str))

    logging.info('')
    return current_split_best_acc

def save_model(net, logs, name):
    torch.save({'logs': logs, 'state_dict': net.state_dict()}, config.save_dir + f'/{name}')

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_args_for_debug():
    args = base_args.BaseArgs()
    args.dataset = "dtd"
    args.seed = 1
    args.gpu_id = 0
    args.epochs = 140
    args.net = "resnet50"
    args.special_aug = "classic"
    args.logdir = f'logs/{args.dataset}/test_delete_me'
    args.train_sample_ratio = 1.0
    args.aug_json = None # "/mnt/raid/home/user_name/datasets/stanford_cars/aug_data/regular/sd_v1.5-SDEdit_strength_0.5/None/ALIA_prompt_w_sub_class/v1-complete_ALIA-res_512-num_2-gs_7.5-num_inf_steps_30_seed_0/semantic_filtering-alia_conf_filtering-aug.json"
    args.aug_sample_ratio = 0.5
    args.few_shot = None
    return args


USE_AMP = True
DONT_WANDB = False
DEBUG = 0
RUN_SCRIPT_HERE = 0
if __name__ == '__main__':
    args = get_args() if not DEBUG else get_args_for_debug()
    args = get_args_for_debug() if DEBUG or RUN_SCRIPT_HERE else args
    if DEBUG:
        # pass CUDA_LAUNCH_BLOCKING=1 for debugging
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        DONT_WANDB = True

    main(args)
