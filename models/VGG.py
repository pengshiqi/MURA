# -*- coding: utf-8 -*-

import math
from .BasicModule import BasicModule
from torch import nn
from torch.nn import functional as F
from torchvision import models


class VGG19(BasicModule):

    def __init__(self, num_classes=2):
        model = models.vgg19(pretrained=True)

        super(VGG19, self).__init__()

        self.features = nn.Sequential(*list(model.features.children()))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 10 * 10, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG16(BasicModule):

    def __init__(self, num_classes=2):
        model = models.vgg16(pretrained=True)

        super(VGG16, self).__init__()

        self.features = nn.Sequential(*list(model.features.children()))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 10 * 10, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

