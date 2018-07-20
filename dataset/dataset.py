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


def logo_filter(data, threshold=200):

    im = Image.new('L', data.size)

    list_data = list(data.split()[0].getdata())

    pixels = [x if x < threshold else 0 for x in list_data]

    im.putdata(data=pixels)

    return im


class MURA_Dataset(object):

    def __init__(self, root, csv_path, part='all', transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据

        train set:     train = True,  test = False
        val set:       train = False, test = False
        test set:      train = False, test = True

        part = 'all', 'XR_HAND', etc.
        用于提取特定部位的数据。
        """

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            if part == 'all':
                imgs = [root + str(x, encoding='utf-8').strip() for x in d]  # 所有图片的存储路径, [:-1]目的是抛弃最末尾的\n
            else:
                imgs = [root + str(x, encoding='utf-8').strip() for x in d if
                        str(x, encoding='utf-8').strip().split('/')[2] == part]

        self.imgs = imgs
        self.train = train
        self.test = test

        if transforms is None:

            if self.train and not self.test:
                # 这里的X光图是1 channel的灰度图
                self.transforms = T.Compose([
                    # T.Lambda(logo_filter),
                    T.Resize(320),
                    T.RandomCrop(320),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(30),
                    T.ToTensor(),
                    T.Lambda(lambda x: t.cat([x[0].unsqueeze(0), x[0].unsqueeze(0), x[0].unsqueeze(0)], 0)),  # 转换成3 channel
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
            if not self.train:
                # 这里的X光图是1 channel的灰度图
                self.transforms = T.Compose([
                    # T.Lambda(logo_filter),
                    T.Resize(320),
                    T.CenterCrop(320),
                    T.ToTensor(),
                    T.Lambda(lambda x: t.cat([x[0].unsqueeze(0), x[0].unsqueeze(0), x[0].unsqueeze(0)], 0)),  # 转换成3 channel
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据：data, label, path, body_part
        """

        img_path = self.imgs[index]

        data = Image.open(img_path)

        data = self.transforms(data)

        # label
        if not self.test:
            label_str = img_path.split('_')[-1].split('/')[0]
            if label_str == 'positive':
                label = 1
            elif label_str == 'negative':
                label = 0
            else:
                print(img_path)
                print(label_str)
                raise IndexError

        if self.test:
            label = 0

        # body part
        body_part = img_path.split('/')[6]

        return data, label, img_path, body_part

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    from config.config import opt
    from tqdm import tqdm
    train_data = MURA_Dataset(opt.data_root, opt.train_image_paths, train=True)
    l = [x[0] for x in tqdm(train_data)]
    x = t.cat(l, 0)
    print(x.mean())
    print(x.std())

