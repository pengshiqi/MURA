# -*- coding: utf-8 -*-

import re
import copy
import torch as t
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Variable

from .BasicModule import BasicModule

# load the original DenseNet model
model = models.densenet169(pretrained=True)
model.cuda()
# model.load_state_dict(t.load('./models/pretrained_models/densenet169-b2777c0a.pth'))


# create custom DenseNet
class DenseNet169(BasicModule):

    def __init__(self, num_classes=2):
        super(DenseNet169, self).__init__()

        # model = models.densenet169(pretrained=False)

        self.features = nn.Sequential(*list(model.features.children()))

        self.classifier = nn.Linear(1664, num_classes)

        self.ada_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # print('out.size():', out.size()) -> torch.Size([8, 1664, 10, 10])
        out = self.ada_pooling(out).view(features.size(0), -1)
        # print('out.size():', out.size()) -> torch.Size([8, 1664])
        out = self.classifier(out)
        # print('out.size():', out.size()) -> torch.Size([8, 2])
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

        # model = models.densenet169(pretrained=False)

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


class MultiBranchDenseNet169(BasicModule):

    def __init__(self, num_classes=2):
        super(MultiBranchDenseNet169, self).__init__()

        # model = models.densenet169(pretrained=False)

        self.features_common = nn.Sequential(
            model.features.conv0,
            model.features.norm0,
            model.features.relu0,
            model.features.pool0,
            model.features.denseblock1,
            model.features.transition1,
            model.features.denseblock2,
            model.features.transition2,
            model.features.denseblock3,
            model.features.transition3
        )

        self.dropout = nn.Dropout(0.5)

        for x in ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']:
            setattr(self, f'features_specific_{x}', copy.deepcopy(nn.Sequential(model.features.denseblock4,
                                                                                model.features.norm5)))
            setattr(self, f'ada_pooling_{x}', nn.AdaptiveAvgPool2d((1, 1)))
            setattr(self, f'classifier_{x}', nn.Linear(1664, num_classes))

        # 查看 self 的所有 attributes
        # print(dir(self))

    def forward(self, x, body_part):
        x = self.features_common(x)
        # print('x.size(): ', x.size()) -> torch.Size([8, 640, 10, 10])

        out1 = Variable(t.FloatTensor())
        for (xx, bp) in zip(x, body_part):
            d = xx.unsqueeze(0).cuda()
            d = getattr(self, f'features_specific_{bp}')(d)
            if out1.size():
                out1 = t.cat([out1, d], dim=0)
            else:
                out1 = d

        # print('out1.size(): ', out1.size()) -> torch.Size([8, 1664, 10, 10])
        out2 = F.relu(out1, inplace=True)
        out2 = self.dropout(0.5)(out2)

        # print('out2.size(): ', out2.size()) -> torch.Size([8, 1664, 10, 10])

        out3 = Variable(t.FloatTensor(), requires_grad=True)
        for (xx, bp) in zip(out2, body_part):
            d = xx.unsqueeze(0).cuda()
            d = getattr(self, f'ada_pooling_{bp}')(d).view(d.size(0), -1)
            if out3.size():
                out3 = t.cat([out3, d], dim=0)
            else:
                out3 = d

        # print('out3.size(): ', out3.size()) -> torch.Size([8, 1664])

        out4 = Variable(t.FloatTensor(), requires_grad=True)
        for (xx, bp) in zip(out3, body_part):
            xx = xx.cuda()
            xx = getattr(self, f'classifier_{bp}')(xx)
            d = xx.unsqueeze(0)
            if out4.size():
                out4 = t.cat([out4, d], dim=0)
            else:
                out4 = d

        # print('out4.size(): ', out4.size()) -> torch.Size([8, 2])

        return out4

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

