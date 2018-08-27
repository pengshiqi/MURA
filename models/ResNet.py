# -*- coding: utf-8 -*-

import torch as t
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Variable

from .BasicModule import BasicModule

model = models.resnet152(pretrained=True)


# create custom DenseNet
class ResNet152(BasicModule):

    def __init__(self, num_classes):
        super(ResNet152, self).__init__()

        self.features = nn.Sequential(model.conv1,
                                      model.bn1,
                                      model.relu,
                                      model.maxpool,
                                      model.layer1,
                                      model.layer2,
                                      model.layer3,
                                      model.layer4)

        self.ada_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.features(x)
        # out = F.relu(features, inplace=True)
        out = self.ada_pooling(features).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def get_config_optim(self, lr, lr_pre):
        return [{'params': self.features.parameters(), 'lr': lr_pre},
                {'params': self.ada_pooling.parameters()},
                {'params': self.classifier.parameters()}]
