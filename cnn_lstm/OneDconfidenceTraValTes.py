import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import argparse
import tensorboardX
import os
import random
import numpy as np
from utils import AverageMeter
from sklearn.metrics import confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101, resnet50
from models import cnnlstm, cnnBilstm


def calculate_accuracy(out, y):
    correct = (out.ge(0.5) == y).sum().item()
    n = y.shape[0]
    return correct / n

# # original CNNLSTM for 1D confidence score.
# class CNNLSTM(nn.Module):
#     def __init__(self, num_classes=2):
#         super(CNNLSTM, self).__init__()
#         self.resnet = resnet18(pretrained=True)
#         self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
#         self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
#         self.fc1 = nn.Linear(256, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#         self.output = nn.Linear(num_classes, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x_3d):
#         hidden = None
#         for t in range(x_3d.size(1)):
#             with torch.no_grad():
#                 x = self.resnet(x_3d[:, t, :, :, :])
#             out, hidden = self.lstm(x.unsqueeze(0), hidden)
#
#         x = self.fc1(out[-1, :, :])
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = self.output(x)
#         x = self.sigmoid(x)
#         return x.squeeze()
#
#
# def generate_model(opt, device):
#     assert opt.model in [
#         'cnnlstm'
#     ]
#
#     if opt.model == 'cnnlstm':
#         model = CNNLSTM(num_classes=opt.n_classes)
#     return model.to(device)

## LSTM
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2, model='Res18'):
        super(CNNLSTM, self).__init__()
        self.model = model
        if model == 'Res18':
            self.resnet = resnet18(pretrained=True)
            self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        if model == 'Res50':
            self.resnet = resnet50(pretrained=True)
            self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        if model == 'Res101':
            self.resnet = resnet101(pretrained=True)
            self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))

        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

        if model == 'CNN':
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=4),
                nn.BatchNorm2d(8),  # 对这16个结果进行规范处理，
                nn.ReLU(),  # 激活函数
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(8 * 18 * 18, 128))
            self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2)
            self.fc1 = nn.Linear(64, 32)
            self.fc2 = nn.Linear(32, num_classes)

        self.output = nn.Linear(num_classes, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            if self.model == 'CNN':
                x = self.layer1(x_3d[:, t, :, :, :])
            else:
                with torch.no_grad():
                    x = self.resnet(x_3d[:, t, :, :, :])
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x.squeeze()

# bi-LSTM
class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes=2, model='Res18'):
        super(CNNBiLSTM, self).__init__()
        self.model = model
        if model == 'Res18':
            self.resnet = resnet18(pretrained=True)
            self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        if model == 'Res50':
            self.resnet = resnet50(pretrained=True)
            self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        if model == 'Res101':
            self.resnet = resnet101(pretrained=True)
            self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))

        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3, bidirectional=True)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

        if model == 'CNN':
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=4),
                nn.BatchNorm2d(8),  # 对这16个结果进行规范处理，
                nn.ReLU(),  # 激活函数
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(8 * 18 * 18, 128))
            self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, bidirectional=True)
            self.fc1 = nn.Linear(128, 32)
            self.fc2 = nn.Linear(32, num_classes)

        self.output = nn.Linear(num_classes, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            if self.model == 'CNN':
                x = self.layer1(x_3d[:, t, :, :, :])
            else:
                with torch.no_grad():
                    x = self.resnet(x_3d[:, t, :, :, :])
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        out = out[-1, :, :]
        if self.model == 'CNN':
            out = torch.cat((out[:, :64], out[:, 64:]), dim=1)
        else:
            out = torch.cat((out[:, :256], out[:, 256:]), dim=1)
        x = self.fc1(out)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x.squeeze()

## Transformer
class CNNTrans(nn.Module):
    def __init__(self, num_classes=2, model='Res18', num_layers=3, num_heads=8, hidden_dim=256, dropout=0.1):
        super(CNNTrans, self).__init__()
        self.model = model
        if model == 'Res18':
            self.resnet = resnet18(pretrained=True)
            self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        if model == 'Res50':
            self.resnet = resnet50(pretrained=True)
            self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        if model == 'Res101':
            self.resnet = resnet101(pretrained=True)
            self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))

        self.embedding = nn.Linear(300, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim, dropout=dropout),
            num_layers
        )
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

        if model == 'CNN':
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=4),
                nn.BatchNorm2d(8),  # 对这16个结果进行规范处理，
                nn.ReLU(),  # 激活函数
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(8 * 18 * 18, 128))

            self.embedding = nn.Linear(128, 64)
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(64, num_heads, dim_feedforward=64, dropout=dropout),
                2
            )
            self.fc1 = nn.Linear(64, 32)
            self.fc2 = nn.Linear(32, num_classes)

        self.output = nn.Linear(num_classes, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x_3d):
        hidden = None
        batch_size, sequence_length, channel, h, w = x_3d.size()
        x = x_3d.view(batch_size * sequence_length, channel, h, w)
        if self.model == 'CNN':
            x = self.layer1(x)
        else:
            with torch.no_grad():
                x = self.resnet(x)
        x = self.embedding(x)
        x = x.view(batch_size, sequence_length, -1)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        out = x

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x.squeeze()
#
def generate_model(opt, device):
    if opt.tempro == 'lstm':
        model = CNNLSTM(num_classes=opt.n_classes, model=opt.model)
    elif opt.tempro == 'bi-lstm':
        model = CNNBiLSTM(num_classes=opt.n_classes, model=opt.model)
    else:
        model = CNNBiLSTM(num_classes=opt.n_classes, model=opt.model)
    return model.to(device)


def train_epoch(model, data_loader, criterion, optimizer, epoch, log_interval, device):
    model.train()

    train_loss = 0.0
    losses = AverageMeter()
    accuracies = AverageMeter()
    for batch_idx, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        if outputs.shape == torch.Size([]):
            outputs = outputs.unsqueeze(0)
        targets = targets.float()
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        train_loss += loss.item()
        losses.update(loss.item(), data.size(0))
        accuracies.update(acc, data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = train_loss / log_interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(data_loader.dataset), 100. * (batch_idx + 1) / len(data_loader),
                avg_loss))
            train_loss = 0.0

    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        len(data_loader.dataset), losses.avg, accuracies.avg * 100))

    return losses.avg, accuracies.avg


def val_epoch(model, data_loader, criterion, device):
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()
    with torch.no_grad():
        for (data, targets) in data_loader:
            data, targets = data.to(device), targets.to(device)
            targets = targets.float()
            outputs = model(data)
            if outputs.shape == torch.Size([]):
                outputs = outputs.unsqueeze(0)

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))

    # show info
    print(
        'Validation set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(len(data_loader.dataset), losses.avg,
                                                                                   accuracies.avg * 100))
    return losses.avg, accuracies.avg


def test_epoch(model, data_loader, criterion, device):
    i = 0
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    distribution = []
    with torch.no_grad():
        for (data, targets) in data_loader:
            data, targets = data.to(device), targets.to(device)
            targets = targets.float()
            outputs = model(data)
            if outputs.shape == torch.Size([]):
                outputs = outputs.unsqueeze(0)

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            score = outputs
            pred = outputs.ge(0.5).int()
            # score, pred = outputs.topk(1, 1, True)
            pred = pred.t().squeeze()
            score = score.squeeze()

            if i == 0:
                score_auc = score
                targets_auc = targets
                pred_CM = pred
            else:
                if score.shape.__len__() == 0:
                    score = score.unsqueeze(0)
                    pred = pred.unsqueeze(0)
                score_auc = torch.cat((score_auc, score))
                targets_auc = torch.cat((targets_auc, targets))
                pred_CM = torch.cat((pred_CM, pred))
            i += 1

            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))

            outputArray = outputs.cpu().detach().numpy()
            outputArray = outputArray.reshape((outputArray.shape[0], 1))
            targetArray = targets.cpu().detach().numpy().reshape((outputArray.shape[0], 1))
            predArray = pred.cpu().detach().numpy().reshape((outputArray.shape[0], 1))
            scoreArray = score.cpu().detach().numpy().reshape((outputArray.shape[0], 1))

            stat = np.concatenate((outputArray, targetArray, predArray, scoreArray), 1)
            distribution.append(stat)

        CM = confusion_matrix(targets_auc.cpu(), pred_CM.cpu())
        tn = CM[0][0]
        tp = CM[1][1]
        fp = CM[0][1]
        fn = CM[1][0]
        sensitivity = (tp / (tp + fn)) * 100
        specificity = (tn / (tn + fp)) * 100
        auc = roc_auc_score(targets_auc.cpu(), score_auc.cpu())
        precision = (tp / (tp + fp)) * 100
        recall = (tp / (tp + fn)) * 100

    # show info
    print('Test set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}\tsensitivity: {:.4f}\tspecificity: {:.4f}'
          '\tauc: {:.4f}\tprecision: {:.4f}\trecall: {:.4f}%'.format(len(data_loader.dataset), losses.avg,
                                                                     accuracies.avg * 100, sensitivity, specificity,
                                                                     auc, precision, recall))
    distribution = np.vstack(distribution)
    return losses.avg, accuracies.avg, sensitivity, specificity, CM, distribution
