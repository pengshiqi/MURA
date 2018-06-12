# -*- coding: utf-8 -*-

import torch as t
from torch import nn
from torch.nn import functional as F
from torchvision import models

from .BasicModule import BasicModule

# load the original DenseNet model
model = models.densenet169(pretrained=True)


# create custom DenseNet
class DenseNet169(BasicModule):

    def __init__(self, num_classes):
        super(DenseNet169, self).__init__()

        self.features = nn.Sequential(*list(model.features.children()))

        self.classifier = nn.Linear(1664, num_classes)

        self.ada_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.ada_pooling(out).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.ada_pooling.parameters()},
                {'params': self.classifier.parameters()}]


# create custom DenseNet
class CustomDenseNet169(BasicModule):

    def __init__(self, num_classes):
        super(CustomDenseNet169, self).__init__()

        self.features = nn.Sequential(*list(model.features.children()))

        # self.classifier = nn.Linear(26624, num_classes)
        self.classifier = nn.Linear(30 * 1664, num_classes)

        self.ada_pooling1 = nn.AdaptiveAvgPool2d((1, 1))
        self.ada_pooling2 = nn.AdaptiveAvgPool2d((2, 2))
        self.ada_pooling3 = nn.AdaptiveAvgPool2d((3, 3))
        self.ada_pooling4 = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out1 = self.ada_pooling1(out).view(features.size(0), -1)
        out2 = self.ada_pooling2(out).view(features.size(0), -1)
        out3 = self.ada_pooling3(out).view(features.size(0), -1)
        out4 = self.ada_pooling4(out).view(features.size(0), -1)
        out = t.cat([out1, out2, out3, out4], 1)

        # print(out.size())
        out = self.classifier(out)
        return out

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.ada_pooling1.parameters()},
                {'params': self.ada_pooling2.parameters()},
                {'params': self.ada_pooling3.parameters()},
                {'params': self.ada_pooling4.parameters()},
                {'params': self.classifier.parameters()}]

