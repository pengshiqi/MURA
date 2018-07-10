# -*- coding: utf-8 -*-

import re
import torch as t
from torch import nn
from torch.nn import functional as F
from torchvision import models

from .BasicModule import BasicModule

# load the original DenseNet model
# model = models.densenet169(pretrained=False)
# model.load_state_dict(t.load('./models/pretrained_models/densenet169-b2777c0a.pth'))


# create custom DenseNet
class DenseNet169(BasicModule):

    def __init__(self, num_classes=2):
        super(DenseNet169, self).__init__()

        model = models.densenet169(pretrained=False)

        self.features = nn.Sequential(*list(model.features.children()))

        self.classifier = nn.Linear(1664, num_classes)

        self.ada_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.ada_pooling(out).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def load(self, path):
        """
        可加载指定路径的模型
        """
        # GPU 加载模型
        # self.load_state_dict(t.load(path))

        # 使用CPU加载GPU模型
        state_dict = t.load(path, map_location=lambda storage, loc: storage)
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        self.load_state_dict(state_dict)


# create custom DenseNet
class CustomDenseNet169(BasicModule):

    def __init__(self, num_classes=2):
        super(CustomDenseNet169, self).__init__()

        model = models.densenet169(pretrained=False)

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

