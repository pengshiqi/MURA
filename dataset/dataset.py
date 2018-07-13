# -*- coding: utf-8 -*-

import numpy as np
import torch as t
from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# training set 的 mean 和 std
# >>> train_data = MURA_Dataset(opt.data_root, opt.train_image_paths, train=True)
# >>> l = [x[0] for x in tqdm(train_data)]
# >>> x = t.cat(l, 0)
# >>> x.mean()
# >>> x.std()
MURA_MEAN = [0.22588661454502146] * 3
MURA_STD = [0.17956269377916526] * 3


# class Rescale():
#     def __init__(self, isrescale):
#         self.isrescale = isrescale
#
#     def __call__(self, x):
#         if self.isrescale:
#             return self.rescale(x)
#         else:
#             return x
#
#     def rescale(self, x):
#         _, h, w = x.size()
#
#         def totensor(in_list):
#             return t.Tensor(np.asarray(in_list).reshape(-1, 1, 1))
#
#         return totensor(IMAGENET_STD) / totensor(MURA_STD) * (x - totensor(MURA_MEAN)) + totensor(IMAGENET_MEAN)


class MURA_Dataset(object):

    def __init__(self, root, csv_path, transforms=None, train=True, test=False, rescale_fg=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据

        test set:  train = x,     test = True
        train set: train = True,  test = False
        val set:   train = False, test = False

        """

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            imgs = [root + str(x, encoding='utf-8')[:-1] for x in d]  # 所有图片的存储路径, [:-1]目的是抛弃最末尾的\n

        self.imgs = imgs
        self.rescale_fg = rescale_fg

        if transforms is None:

            if train:
                # 这里的X光图是1 channel的灰度图
                self.transforms = T.Compose([
                    T.Resize(320),
                    T.RandomCrop(320),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(30),
                    T.ToTensor(),
                    T.Lambda(lambda x: t.cat([x[0].unsqueeze(0), x[0].unsqueeze(0), x[0].unsqueeze(0)], 0)),  # 转换成3 channel
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
            if test:
                # 这里的X光图是1 channel的灰度图
                self.transforms = T.Compose([
                    T.Resize(320),
                    T.CenterCrop(320),
                    T.ToTensor(),
                    T.Lambda(lambda x: t.cat([x[0].unsqueeze(0), x[0].unsqueeze(0), x[0].unsqueeze(0)], 0)),  # 转换成3 channel
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据：data, label, path
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

        return data, label, img_path

    def __len__(self):
        return len(self.imgs)


class MURAClass_Dataset(object):

    def __init__(self, root, csv_path, part, transforms=None, train=True, test=False, rescale_fg=False):
        with open(csv_path, 'rb') as F:
            d = F.readlines()
            if part == 'all':
                imgs = [root + str(x, encoding='utf-8').strip() for x in d]  # 所有图片的存储路径, [:-1]目的是抛弃最末尾的\n
            else:
                imgs = [root + str(x, encoding='utf-8').strip() for x in d if str(x, encoding='utf-8').strip().split('/')[2].split('_')[1]==part]

        self.imgs = imgs
        self.rescale_fg = rescale_fg

        if transforms is None:

            if train:
                # 这里的X光图是1 channel的灰度图
                self.transforms = T.Compose([
                    T.Resize(320),
                    T.RandomCrop(320),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(30),
                    T.ToTensor(),
                    T.Lambda(lambda x: t.cat([x[0].unsqueeze(0), x[0].unsqueeze(0), x[0].unsqueeze(0)], 0)),  # 转换成3 channel
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
            if test:
                # 这里的X光图是1 channel的灰度图
                self.transforms = T.Compose([
                    T.Resize(320),
                    T.CenterCrop(320),
                    T.ToTensor(),
                    T.Lambda(lambda x: t.cat([x[0].unsqueeze(0), x[0].unsqueeze(0), x[0].unsqueeze(0)], 0)),  # 转换成3 channel
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据：data, label, body_part, path
        """

        img_path = self.imgs[index]

        label_str = img_path.split('_')[-1].split('/')[0]
        if label_str == 'positive':
            label = 1
        elif label_str == 'negative':
            label = 0
        else:
            raise IndexError

        body_part = img_path.split('/')[6].split('_')[1]
        if body_part == 'ELBOW':
            body_part = 1
        elif body_part == 'FINGER':
            body_part = 2
        elif body_part == 'FOREARM':
            body_part = 3
        elif body_part == 'HAND':
            body_part = 4
        elif body_part == 'HUMERUS':
            body_part = 5
        elif body_part == 'SHOULDER':
            body_part = 6
        elif body_part == 'WRIST':
            body_part = 7
        else:
            print(body_part)
            raise IndexError

        data = Image.open(img_path)

        data = self.transforms(data)

        return data, label, body_part, img_path

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    # from config import opt
    from tqdm import tqdm
    train_data = MURAClass_Dataset('/DATA4_DB3/data/public/', '/DATA4_DB3/data/public/MURA-v1.1/train_image_paths.csv', part='all', train=True)
    # print(train_data[2][0].size())
    print(train_data[0])
    # l = [x[0] for x in tqdm(train_data)]
    # x = t.cat(l, 0)
    # print(x.mean())
    # print(x.std())

