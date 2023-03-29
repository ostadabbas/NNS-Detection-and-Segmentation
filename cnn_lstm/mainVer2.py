import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX
import os
import random
import numpy as np

from train import train_epoch
from torch.utils.data import DataLoader
from validation import val_epoch, test_epoch
from opts import parse_opts
from model import generate_model
from torch.optim import lr_scheduler
from dataset import get_training_set, get_validation_set
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from torch.utils.tensorboard import SummaryWriter


def resume_model(opt, model, optimizer):
    """ Resume model
	"""
    checkpoint = torch.load(opt.resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model Restored from Epoch {}".format(checkpoint['epoch']))
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch


def get_loaders(opt):
    """ Make dataloaders for train and validation sets
	"""
    # train loader
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    spatial_transform = Compose([
        # crop_method,
        Scale((opt.sample_size, opt.sample_size)),
        # RandomHorizontalFlip(),
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = TemporalRandomCrop(16)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform,
                                     temporal_transform, target_transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True)

    # validation loader
    spatial_transform = Compose([
        Scale((opt.sample_size, opt.sample_size)),
        # CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value), norm_method
    ])
    target_transform = ClassLabel()
    temporal_transform = LoopPadding(16)
    validation_data = get_validation_set(
        opt, spatial_transform, temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True)
    return train_loader, val_loader


def get_loaders(opt):
    """ Make dataloaders for train and validation sets
	"""
    # train loader
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    spatial_transform = Compose([
        # crop_method,
        Scale((opt.sample_size, opt.sample_size)),
        # RandomHorizontalFlip(),
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = TemporalRandomCrop(16)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform,
                                     temporal_transform, target_transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True)

    # validation loader
    spatial_transform = Compose([
        Scale((opt.sample_size, opt.sample_size)),
        # CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value), norm_method
    ])
    target_transform = ClassLabel()
    temporal_transform = LoopPadding(16)
    validation_data = get_validation_set(
        opt, spatial_transform, temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True)
    return train_loader, val_loader


def get_test_loaders(opt):
    """ Make dataloaders for test sets
	"""
    # train loader
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    spatial_transform = Compose([
        # crop_method,
        Scale((opt.sample_size, opt.sample_size)),
        # RandomHorizontalFlip(),
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = TemporalRandomCrop(16)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform,
                                     temporal_transform, target_transform)
    validation_data = get_validation_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
    test_dataset = torch.utils.data.ConcatDataset([training_data, validation_data])
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True)

    return test_loader


def main_worker(opt):
    # writer = SummaryWriter('./path/to/log')

    # opt = parse_opts()
    # print(opt)

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA for PyTorch
    device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")

    # tensorboard
    summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs')

    # defining model
    model = generate_model(opt, device)
    # get data loaders
    train_loader, val_loader = get_loaders(opt)

    opt.annotation_path = './opticalFlowDataset/test_dataset/R3/annotation/ucf101_01.json'
    opt.video_path = './opticalFlowDataset/test_dataset/R3/image_data/'

    test_loader = get_test_loaders(opt)

    opt.annotation_path = './opticalFlowDataset/training_dataset_mix/mix_no3/annotation/ucf101_01.json'
    opt.video_path = './opticalFlowDataset/training_dataset_mix/mix_no3/image_data/'

    # optimizer
    crnn_params = list(model.parameters())
    optimizer = torch.optim.Adam(crnn_params, lr=opt.lr_rate, weight_decay=opt.weight_decay)

    # scheduler = lr_scheduler.ReduceLROnPlateau(
    # 	optimizer, 'min', patience=opt.lr_patience)

    criterion = nn.CrossEntropyLoss()

    # resume model
    if opt.resume_path:
        start_epoch = resume_model(opt, model, optimizer)
    else:
        start_epoch = 1

    # start training
    for epoch in range(start_epoch, opt.n_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch, opt.log_interval, device)
        val_loss, val_acc = val_epoch(
            model, val_loader, criterion, device)
        test_loss, test_acc, sensitivity, specificity, CM, distribution = test_epoch(
            model, test_loader, criterion, device)

        # for score visualization matrix
        # filename = str(epoch) + '.npy'
        # with open(filename, 'wb') as f:
        #     np.save(f, distribution)

        # saving weights to checkpoint
        if (epoch) % opt.save_interval == 0:
            # scheduler.step(val_loss)
            # write summary
            summary_writer.add_scalar(
                'losses/train_loss', train_loss, global_step=epoch)
            summary_writer.add_scalar(
                'losses/val_loss', val_loss, global_step=epoch)
            summary_writer.add_scalar(
                'losses/test_loss', test_loss, global_step=epoch)
            summary_writer.add_scalar(
                'acc/train_acc', train_acc * 100, global_step=epoch)
            summary_writer.add_scalar(
                'acc/val_acc', val_acc * 100, global_step=epoch)
            summary_writer.add_scalar(
                'acc/test_acc', test_acc * 100, global_step=epoch)
            summary_writer.add_scalar(
                'statistic/sensitivity', sensitivity, global_step=epoch)
            summary_writer.add_scalar(
                'statistic/specificity', specificity, global_step=epoch)
            tn = CM[0][0]
            tp = CM[1][1]
            fp = CM[0][1]
            fn = CM[1][0]
            precision = (tp/(tp+fp))*100
            recall = (tp/(tp+fn))*100
            # summary_writer.add_scalar(
            #     'statistic/AUC', auc, global_step=epoch)
            summary_writer.add_scalar(
                'statistic/precision', precision, global_step=epoch)
            summary_writer.add_scalar(
                'statistic/recall', recall, global_step=epoch)

            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state, os.path.join('snapshots', f'{opt.model}-Epoch-{epoch}-Loss-{val_loss}.pth'))
            print("Epoch {} model saved!\n".format(epoch))


class augment():
    def __init__(self):
        self.use_cuda = True
        self.gpu = 0
        self.batch_size = 8
        self.n_epochs = 20
        self.num_workers = 0
        self.annotation_path = './opticalFlowDataset/training_dataset_mix/mix_no3/annotation/ucf101_01.json'
        self.video_path = './opticalFlowDataset/training_dataset_mix/mix_no3/image_data/'
        self.dataset = 'ucf101'
        self.sample_size = 150
        self.lr_rate = 0.0001
        self.n_classes = 2

        self.root_path = '/root/data/ActivityNet'
        self.sample_duration = 16
        self.n_val_samples = 1
        self.log_interval = 10
        self.save_interval = 2
        self.model = 'CNN_LSTM'
        self.momentum = 0.9
        self.weight_decay = 0.001
        self.no_mean_norm = False
        self.mean_dataset = 'activitynet'
        self.std_norm = False
        self.nesterov = False
        self.optimizer = 'sgd'
        self.lr_patience = 10
        self.start_epoch = 1
        self.resume_path = ''
        self.pretrain_path = ''
        self.norm_value = 1


if __name__ == "__main__":
    args = augment()
    main_worker(args)

