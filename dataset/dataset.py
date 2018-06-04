# -*- coding: utf-8 -*-

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
MURA_MEAN = [0.22588661454502146]
MURA_STD = [0.17956269377916526]


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
            imgs = [root + str(x, encoding='utf-8')[:-1] for x in d]  # 所有图片的存储路径, [:-1]目的是抛弃最末尾的\n

        imgs_num = len(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if transforms is None:
            normalize = T.Normalize(mean=MURA_MEAN, std=MURA_STD)

            # 这里的X光图是1 channel的灰度图
            self.transforms = T.Compose([
                T.Resize(224),  # 将输入图像的短边resize到这个int数，长边则根据对应比例调整，图像的长宽比不变。
                T.CenterCrop(224),  # 以输入图的中心点为中心点做指定size的crop操作，切出来的图片是正方形
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
        # 因为 T.ToTensor() 的结果是3 channel的，所以取1channel然后unsqueeze(0)
        data = self.transforms(data)[0].unsqueeze(0)
        # data = self.transforms(data)

        return data, label

    def __len__(self):
        return len(self.imgs)
