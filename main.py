# -*- coding: utf-8 -*-

import os
import torch as t
from torch.autograd import Variable
from torchvision.models import densenet169
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import opt
from utils import Visualizer
from dataset import MURA_Dataset


def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # step 1: configure model
    model = densenet169(pretrained=True)
    if opt.use_gpu:
        model.cuda()

    # step 2: data
    train_data = MURA_Dataset(opt.data_root, opt.train_image_paths, train=True)
    val_data = MURA_Dataset(opt.data_root, opt.test_image_paths, train=False)

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # step 3: criterion and optimizer

    # step 4: meters

    # step 5: train


def val(**kwargs):
    pass

def test(**kwargs):
    pass

def help(**kwargs):
    """
        打印帮助的信息： python main.py help
        """

    print("""
        usage : python main.py <function> [--args=value]
        <function> := train | test | help
        example: 
                python {0} train --env='env_MURA' --lr=0.001
                python {0} test --dataset='/path/to/dataset/root/'
                python {0} help
        avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire

    fire.Fire()
