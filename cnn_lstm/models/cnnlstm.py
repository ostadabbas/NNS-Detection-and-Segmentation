import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x


# class CNNLSTM(nn.Module):
#     def __init__(self, num_classes=2):
#         super(CNNLSTM, self).__init__()
#         # self.resnet = resnet101(pretrained=True)
#         # self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=4),
#             nn.BatchNorm2d(8),  # 对这16个结果进行规范处理，
#             nn.ReLU(),  # 激活函数
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Flatten(),
#             nn.Linear(8*18*18, 128))
#         # self.fc_cnn = nn.Linear(8*18*18, 128)
#         self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2)
#         self.fc1 = nn.Linear(64, 32)
#         self.fc2 = nn.Linear(32, num_classes)
#
#     def forward(self, x_3d):
#         hidden = None
#         for t in range(x_3d.size(1)):
#             # with torch.no_grad():
#             x = self.layer1(x_3d[:, t, :, :, :])
#             # print(x.size())
#             # x = self.fc_cnn(x)
#             out, hidden = self.lstm(x.unsqueeze(0), hidden)
#
#         x = self.fc1(out[-1, :, :])
#         x = F.relu(x)
#         x = self.fc2(x)
#         return x