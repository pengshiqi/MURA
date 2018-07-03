# -*- coding: utf-8 -*-

import torch as t
import time
import re


class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        # self.model_name = str(type(self))  # 默认名字
        self.model_name = self.__class__.__name__

    def load(self, path):
        """
        可加载指定路径的模型
        """
        # GPU 加载模型
        self.load_state_dict(t.load(path))

        # 使用CPU加载GPU模型
        # state_dict = t.load(path, map_location=lambda storage, loc: storage)
        # pattern = re.compile(
        #     r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        # for key in list(state_dict.keys()):
        #     res = pattern.match(key)
        #     if res:
        #         new_key = res.group(1) + res.group(2)
        #         state_dict[new_key] = state_dict[key]
        #         del state_dict[key]
        # self.load_state_dict(state_dict)

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name


class Flat(t.nn.Module):
    """
    把输入reshape成（batch_size,dim_length）
    """

    def __init__(self):
        super(Flat, self).__init__()
        # self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)
