# -*- coding: utf-8 -*-

import torch as t
import numpy as np
import copy

from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Variable

from .BasicModule import BasicModule

res152 = models.resnet152(pretrained=True)
dense169 = models.densenet169(pretrained=True)


# create custom DenseNet
class MultiResolutionNet(BasicModule):

    def __init__(self, num_classes):
        super(MultiResolutionNet, self).__init__()

        # self.features = nn.Sequential(model.conv1,
        #                               model.bn1,
        #                               model.relu,
        #                               model.maxpool,
        #                               model.layer1,
        #                               model.layer2,
        #                               model.layer3,
        #                               model.layer4)

        self.prenet = nn.Sequential(res152.conv1, res152.bn1, res152.relu, res152.maxpool)

        self.reslayer1 = res152.layer1
        self.reslayer2 = res152.layer2
        self.reslayer3 = res152.layer3
        self.reslayer4 = res152.layer4

        self.densblock1_1 = dense169.features.denseblock2
        self.densblock1_2 = dense169.features.denseblock3
        self.densblock1_3 = dense169.features.denseblock4

        self.densblock2_1 = copy.deepcopy(dense169.features.denseblock2)
        self.densblock2_2 = copy.deepcopy(dense169.features.denseblock3)
        self.densblock2_3 = copy.deepcopy(dense169.features.denseblock4)

        self.densblock3_1 = copy.deepcopy(dense169.features.denseblock2)
        self.densblock3_2 = copy.deepcopy(dense169.features.denseblock3)
        self.densblock3_3 = copy.deepcopy(dense169.features.denseblock4)

        self.trans1_1 = self.translayer(256, 128)
        self.trans1_2 = self.translayer(1664 + 512, 256)
        self.trans1_3 = self.translayer(1280, 640)

        self.trans2_1 = self.translayer(512, 128)
        self.trans2_2 = self.translayer(1664 + 512, 256)
        self.trans2_3 = self.translayer(1280, 640)

        self.trans3_1 = self.translayer(1024, 128)
        self.trans3_2 = self.translayer(2048 + 512, 256)
        self.trans3_3 = self.translayer(1280, 640)

        self.trans_final = self.translayer(1664, 2)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.ada_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        # print('input', x.size())
        out = self.prenet(x)
        # print('prenet', out.size())
        res_out_1 = self.reslayer1(out)
        # print('reslayer1', res_out_1.size())
        res_out_2 = self.reslayer2(res_out_1)
        # print('reslayer2', res_out_2.size())
        res_out_3 = self.reslayer3(res_out_2)
        # print('reslayer3', res_out_3.size())
        res_out_4 = self.reslayer4(res_out_3)
        # print('reslayer4', res_out_4.size())

        dense_out_1 = self.densblock1_1(self.trans1_1(res_out_1))
        dense_out_2 = self.densblock2_1(self.trans2_1(res_out_2))
        dense_out_3 = self.densblock3_1(self.trans3_1(res_out_3))
        # print('dense1', dense_out_1.size())
        # print('dense2', dense_out_2.size())
        # print('dense3', dense_out_3.size())

        up4to3 = self.upsample3(res_out_4)
        # print('up4to3', up4to3.size())
        dense_out_3 = t.cat([dense_out_3, up4to3], 1)
        dense_out_3 = self.densblock3_2(self.trans3_2(dense_out_3))
        dense_out_3 = self.densblock3_3(self.trans3_3(dense_out_3))
        # print('dense_out_3', dense_out_3.size())

        up3to2 = self.upsample2(dense_out_3)
        # print('up3to2', up3to2.size())
        dense_out_2 = t.cat([dense_out_2, up3to2], 1)
        dense_out_2 = self.densblock2_2(self.trans2_2(dense_out_2))
        dense_out_2 = self.densblock2_3(self.trans2_3(dense_out_2))
        # print('dense_out_2', dense_out_2.size())

        up2to1 = self.upsample1(dense_out_2)
        # print('up2to1', up2to1.size())
        dense_out_1 = t.cat([dense_out_1, up2to1], 1)
        dense_out_1 = self.densblock1_2(self.trans1_2(dense_out_1))
        dense_out_1 = self.densblock1_3(self.trans1_3(dense_out_1))
        # print('dense_out_1', dense_out_1.size())

        out = F.sigmoid(self.trans_final(dense_out_1))
        out = self.ada_pooling(out).view(out.size(0), -1)
        # print('adapooling', out.size())

        return out

    def get_config_optim(self, lr, lr_pre):
        return [{'params': self.prenet.parameters(), 'lr': lr_pre},
                {'params': self.reslayer1.parameters(), 'lr': lr_pre},
                {'params': self.reslayer2.parameters(), 'lr': lr_pre},
                {'params': self.reslayer3.parameters(), 'lr': lr_pre},
                {'params': self.reslayer4.parameters(), 'lr': lr_pre},
                {'params': self.densblock1_1.parameters(), 'lr': lr_pre},
                {'params': self.densblock1_2.parameters(), 'lr': lr_pre},
                {'params': self.densblock1_3.parameters(), 'lr': lr_pre},
                {'params': self.densblock2_1.parameters(), 'lr': lr_pre},
                {'params': self.densblock2_2.parameters(), 'lr': lr_pre},
                {'params': self.densblock2_3.parameters(), 'lr': lr_pre},
                {'params': self.densblock3_1.parameters(), 'lr': lr_pre},
                {'params': self.densblock3_2.parameters(), 'lr': lr_pre},
                {'params': self.densblock3_3.parameters(), 'lr': lr_pre},
                {'params': self.trans1_1.parameters()},
                {'params': self.trans1_2.parameters()},
                {'params': self.trans1_3.parameters()},
                {'params': self.trans2_1.parameters()},
                {'params': self.trans2_2.parameters()},
                {'params': self.trans2_3.parameters()},
                {'params': self.trans3_1.parameters()},
                {'params': self.trans3_2.parameters()},
                {'params': self.trans3_3.parameters()},
                {'params': self.trans_final.parameters()},
                {'params': self.upsample1.parameters()},
                {'params': self.upsample2.parameters()},
                {'params': self.upsample3.parameters()},
                {'params': self.ada_pooling.parameters()}]

    def translayer(self, input, output):
        return nn.Sequential(nn.BatchNorm2d(input),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(input, output, kernel_size=1, stride=1, bias=False))
