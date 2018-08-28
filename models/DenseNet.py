# -*- coding: utf-8 -*-

import torch as t
import numpy as np

from torch import nn
from torch.nn import functional
from torchvision import models

from .BasicModule import BasicModule

# load the original DenseNet model
model = models.densenet169(pretrained=True)


# create custom DenseNet
class DenseNet169(BasicModule):

    def __init__(self, num_classes):
        super(DenseNet169, self).__init__()

        self.features = nn.Sequential(*list(model.features.children()))

        self.ada_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(1664, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = functional.relu(features, inplace=True)
        out = self.ada_pooling(out).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def get_config_optim(self, lr, lr_pre):
        return [{'params': self.features.parameters(), 'lr': lr_pre},
                {'params': self.ada_pooling.parameters()},
                {'params': self.classifier.parameters()}]


# create custom DenseNet
class CustomDenseNet169(BasicModule):

    def __init__(self, num_classes):
        super(CustomDenseNet169, self).__init__()

        self.features = nn.Sequential(*list(model.features.children()))

        self.ada_pooling1 = nn.AdaptiveAvgPool2d((1, 1))
        self.ada_pooling2 = nn.AdaptiveAvgPool2d((2, 2))
        self.ada_pooling3 = nn.AdaptiveAvgPool2d((3, 3))
        self.ada_pooling4 = nn.AdaptiveAvgPool2d((4, 4))

        # self.classifier = nn.Linear(26624, num_classes)
        self.classifier = nn.Linear(30 * 1664, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = functional.relu(features, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out1 = self.ada_pooling1(out).view(features.size(0), -1)
        out2 = self.ada_pooling2(out).view(features.size(0), -1)
        out3 = self.ada_pooling3(out).view(features.size(0), -1)
        out4 = self.ada_pooling4(out).view(features.size(0), -1)
        out = t.cat([out1, out2, out3, out4], 1)

        # print(out.size())
        out = self.classifier(out)
        return out

    def get_config_optim(self, lr, lr_pre):
        return [{'params': self.features.parameters(), 'lr': lr_pre},
                {'params': self.ada_pooling1.parameters()},
                {'params': self.ada_pooling2.parameters()},
                {'params': self.ada_pooling3.parameters()},
                {'params': self.ada_pooling4.parameters()},
                {'params': self.classifier.parameters()}]


# create custom DenseNet
class MultiDenseNet169(BasicModule):

    def __init__(self, num_classes):
        super(MultiDenseNet169, self).__init__()

        self.features = nn.Sequential(*list(model.features.children()))

        self.block1 = nn.Sequential(nn.Conv2d(1664, 1664, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1664),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1664, 1664, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1664),
                                    nn.ReLU(inplace=True))

        self.block2 = nn.Sequential(nn.Conv2d(1664, 1664, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1664),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1664, 1664, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1664),
                                    nn.ReLU(inplace=True))

        self.block3 = nn.Sequential(nn.Conv2d(1664, 1664, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1664),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1664, 1664, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1664),
                                    nn.ReLU(inplace=True))

        self.block4 = nn.Sequential(nn.Conv2d(1664, 1664, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1664),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1664, 1664, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1664),
                                    nn.ReLU(inplace=True))

        self.block5 = nn.Sequential(nn.Conv2d(1664, 1664, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1664),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1664, 1664, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1664),
                                    nn.ReLU(inplace=True))

        self.block6 = nn.Sequential(nn.Conv2d(1664, 1664, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1664),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1664, 1664, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1664),
                                    nn.ReLU(inplace=True))

        self.block7 = nn.Sequential(nn.Conv2d(1664, 1664, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1664),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1664, 1664, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1664),
                                    nn.ReLU(inplace=True))

        self.ada_pooling1 = nn.AdaptiveAvgPool2d((1, 1))
        self.ada_pooling2 = nn.AdaptiveAvgPool2d((2, 2))
        self.ada_pooling3 = nn.AdaptiveAvgPool2d((3, 3))
        self.ada_pooling4 = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier1 = nn.Linear(30 * 1664, num_classes)   # 30 = 1*1 + 2*2 + 3*3 + 4*4
        self.classifier2 = nn.Linear(30 * 1664, num_classes)
        self.classifier3 = nn.Linear(30 * 1664, num_classes)
        self.classifier4 = nn.Linear(30 * 1664, num_classes)
        self.classifier5 = nn.Linear(30 * 1664, num_classes)
        self.classifier6 = nn.Linear(30 * 1664, num_classes)
        self.classifier7 = nn.Linear(30 * 1664, num_classes)

    def forward(self, x, body_part):
        features = self.features(x)
        out = functional.relu(features, inplace=True)
        final_out = []
        for f, bp in zip(out, body_part):
            if int(bp) == 1:
                result = self.block1(t.unsqueeze(f, 0))
                result1 = self.ada_pooling1(result).view(result.size(0), -1)
                result2 = self.ada_pooling2(result).view(result.size(0), -1)
                result3 = self.ada_pooling3(result).view(result.size(0), -1)
                result4 = self.ada_pooling4(result).view(result.size(0), -1)
                result = t.cat([result1, result2, result3, result4], 1)
                result = self.classifier1(result)
            elif int(bp) == 2:
                result = self.block2(t.unsqueeze(f, 0))
                result1 = self.ada_pooling1(result).view(result.size(0), -1)
                result2 = self.ada_pooling2(result).view(result.size(0), -1)
                result3 = self.ada_pooling3(result).view(result.size(0), -1)
                result4 = self.ada_pooling4(result).view(result.size(0), -1)
                result = t.cat([result1, result2, result3, result4], 1)
                result = self.classifier2(result)
            elif int(bp) == 3:
                result = self.block3(t.unsqueeze(f, 0))
                result1 = self.ada_pooling1(result).view(result.size(0), -1)
                result2 = self.ada_pooling2(result).view(result.size(0), -1)
                result3 = self.ada_pooling3(result).view(result.size(0), -1)
                result4 = self.ada_pooling4(result).view(result.size(0), -1)
                result = t.cat([result1, result2, result3, result4], 1)
                result = self.classifier3(result)
            elif int(bp) == 4:
                result = self.block4(t.unsqueeze(f, 0))
                result1 = self.ada_pooling1(result).view(result.size(0), -1)
                result2 = self.ada_pooling2(result).view(result.size(0), -1)
                result3 = self.ada_pooling3(result).view(result.size(0), -1)
                result4 = self.ada_pooling4(result).view(result.size(0), -1)
                result = t.cat([result1, result2, result3, result4], 1)
                result = self.classifier4(result)
            elif int(bp) == 5:
                result = self.block5(t.unsqueeze(f, 0))
                result1 = self.ada_pooling1(result).view(result.size(0), -1)
                result2 = self.ada_pooling2(result).view(result.size(0), -1)
                result3 = self.ada_pooling3(result).view(result.size(0), -1)
                result4 = self.ada_pooling4(result).view(result.size(0), -1)
                result = t.cat([result1, result2, result3, result4], 1)
                result = self.classifier5(result)
            elif int(bp) == 6:
                result = self.block6(t.unsqueeze(f, 0))
                result1 = self.ada_pooling1(result).view(result.size(0), -1)
                result2 = self.ada_pooling2(result).view(result.size(0), -1)
                result3 = self.ada_pooling3(result).view(result.size(0), -1)
                result4 = self.ada_pooling4(result).view(result.size(0), -1)
                result = t.cat([result1, result2, result3, result4], 1)
                result = self.classifier6(result)
            elif int(bp) == 7:
                result = self.block7(t.unsqueeze(f, 0))
                result1 = self.ada_pooling1(result).view(result.size(0), -1)
                result2 = self.ada_pooling2(result).view(result.size(0), -1)
                result3 = self.ada_pooling3(result).view(result.size(0), -1)
                result4 = self.ada_pooling4(result).view(result.size(0), -1)
                result = t.cat([result1, result2, result3, result4], 1)
                result = self.classifier7(result)
            else:
                print('Error Index:', body_part)
                raise IndexError
            final_out.append(result.squeeze_(0))
        final_out = t.stack(final_out)
        return final_out

    def get_config_optim(self, lr, lr_pre):
        return [{'params': self.features.parameters(), 'lr': lr_pre},
                {'params': self.block1.parameters()},
                {'params': self.block2.parameters()},
                {'params': self.block3.parameters()},
                {'params': self.block4.parameters()},
                {'params': self.block5.parameters()},
                {'params': self.block6.parameters()},
                {'params': self.block7.parameters()},
                {'params': self.ada_pooling1.parameters()},
                {'params': self.ada_pooling2.parameters()},
                {'params': self.ada_pooling3.parameters()},
                {'params': self.ada_pooling4.parameters()},
                {'params': self.classifier1.parameters()},
                {'params': self.classifier2.parameters()},
                {'params': self.classifier3.parameters()},
                {'params': self.classifier4.parameters()},
                {'params': self.classifier5.parameters()},
                {'params': self.classifier6.parameters()},
                {'params': self.classifier7.parameters()}]

