# -*- coding: utf-8 -*-

from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MURA_Dataset(object):

    def __init__(self, root, csv_path, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据

        test set:  train = x,     test = True
        train set: train = True,  test = False
        val set:   train = False, test = False

        """
        self.test = test

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            imgs = [root + x for x in d]  # 所有图片的存储路径

        imgs_num = len(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if transforms is None:
            normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

            # TODO
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """

        img_path = self.imgs[index]

        label_str = img_path.split('_')[-1].split('/')[0]
        if label_str == 'positive':
            label = 1
        elif label_str == 'negative':
            label = 0
        else:
            raise IndexError

        data = Image.open(img_path)
        data = self.transforms(data)

        return data, label

    def __len__(self):
        return len(self.imgs)
