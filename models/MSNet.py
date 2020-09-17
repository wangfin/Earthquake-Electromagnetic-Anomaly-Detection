#!/usr/bin/env python
# @Time    : 2020/9/1 16:50
# @Author  : wb
# @File    : MSNet.py

# 多尺度网络

import torch
from torch import nn
from.BasicModule import BasicModule
import torchsnooper

class MSNet(BasicModule):

    # @torchsnooper.snoop()
    def __init__(self, num_classes=2):
        '''
        构建模型架构
        :param num_classes: 全连接层的层数
        '''
        super(MSNet, self).__init__()
        self.model_name = 'MSNet'

        # 第一个分支，层数多，表示局部视野
        self.S1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2, 1)
        )

        self.S1_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2, 1)
        )

        self.S1_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2, 1)
        )

        self.S1_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2, 1)
        )

        self.S1_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2, 1)
        )
        # 第二个分支
        # S2_1 = S1_2
        self.S2_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=120, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2, 1)
        )
        self.S2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=31, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2, 1)
        )
        # 第三个分支
        self.S3_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=120, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2, 1)
        )

        # 分类用的全连接
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    # @torchsnooper.snoop()
    def forward(self, x):
        # 构建第一分支
        S1_1_out = self.S1_1(x)
        S2_1_out = self.S2_1(x)
        S3_1_out = self.S3_1(x)

        S1_2_out = self.S1_2(S1_1_out)

        # S1_3_in = S1_2_out + S2_1_out
        S1_3_in = torch.cat([S1_2_out, S2_1_out], 1)
        S1_3_out = self.S1_3(S1_3_in)

        S1_4_out = self.S1_4(S1_3_out)

        # S2_2_in = S2_1_out + S3_1_out
        S2_2_in = torch.cat([S2_1_out, S3_1_out], 1)
        S2_2_out = self.S2_2(S2_2_in)

        # S1_5_in = S1_4_out + S2_1_out
        S1_5_in = torch.cat([S1_4_out, S2_2_out], 1)
        S1_5_out = self.S1_5(S1_5_in)

        # 全局平均池化
        S1_out = nn.functional.adaptive_avg_pool2d(S1_5_out, (1, 1))
        S2_out = nn.functional.adaptive_avg_pool2d(S2_2_out, (1, 1))
        S3_out = nn.functional.adaptive_avg_pool2d(S3_1_out, (1, 1))

        # 展平层
        S1_out = S1_out.view(S1_out.size(0), -1)
        S2_out = S1_out.view(S2_out.size(0), -1)
        S3_out = S3_out.view(S3_out.size(0), -1)
        # S2_out = torch.flatten(S2_out)
        # S3_out = torch.flatten(S3_out)

        # 合并特征
        S_out = torch.cat([S1_out, S2_out, S3_out], 1)

        # 全连接层
        y = self.fc(S_out)

        return y


