# -*- coding: utf-8 -*-

import math
import copy
import torch as t
from .BasicModule import BasicModule
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Variable


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


class MultiBranchVGG19(BasicModule):

    def __init__(self, num_classes=2):
        model = models.vgg19(pretrained=True)
        model.cuda()

        super(MultiBranchVGG19, self).__init__()

        # 0 - 27 层共用
        self.features_shared = nn.Sequential(*list(model.features.children())[:28])

        # 28 - 36 层和classifier 分开训练
        for x in ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']:
            setattr(self, f'features_specific_{x}', copy.deepcopy(nn.Sequential(*list(model.features.children())[28:])))
            setattr(self, f'classifier_{x}', nn.Sequential(
                                                nn.Linear(512 * 10 * 10, 4096),
                                                nn.ReLU(True),
                                                nn.Dropout(),
                                                # nn.Linear(4096, 4096),
                                                copy.deepcopy(model.classifier[3]),
                                                nn.ReLU(True),
                                                nn.Dropout(),
                                                nn.Linear(4096, num_classes),
                                            ))

    def forward(self, x, body_part):
        x = self.features_shared(x)

        out1 = Variable(t.FloatTensor())
        for (xx, bp) in zip(x, body_part):
            d = xx.unsqueeze(0).cuda()
            d = getattr(self, f'features_specific_{bp}')(d)
            if out1.size():
                out1 = t.cat([out1, d], dim=0)
            else:
                out1 = d

        out1 = out1.view(out1.size(0), -1)

        out2 = Variable(t.FloatTensor(), requires_grad=True)
        for (xx, bp) in zip(out1, body_part):
            xx = xx.cuda()
            xx = getattr(self, f'classifier_{bp}')(xx)
            d = xx.unsqueeze(0)
            if out2.size():
                out2 = t.cat([out2, d], dim=0)
            else:
                out2 = d

        return out2
