# -*- coding: utf-8 -*-

from torch import nn
from torch.nn import functional as F
from torchvision import models

from .BasicModule import BasicModule

# load the original DenseNet model
model = models.densenet169(pretrained=True)


# create custom DenseNet
class CustomDenseNet169(BasicModule):

    def __init__(self, num_classes):
        super(CustomDenseNet169, self).__init__()

        self.features = nn.Sequential(*list(model.features.children()))

        # self.classifier = nn.Linear(26624, num_classes)
        self.classifier = nn.Linear(1664, num_classes)

        self.ada_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.ada_pooling(out).view(features.size(0), -1)
        # print(out.size())
        out = self.classifier(out)
        return out

