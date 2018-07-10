# -*- coding: utf-8 -*-

import warnings
import time


class Config(object):
    use_visdom = True
    env = 'MURA'                                                    # visdom 环境
    model = 'DenseNet169'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    # 组合模型的 模型类型 和 路径
    ensemble_model_types = ['DenseNet169', 'ResNet152', 'VGG16']
    ensemble_model_paths = ['checkpoints/best_densenet169_0702.pth',
                            'checkpoints/best_resnet152_0708.pth',
                            'checkpoints/best_vgg16_0708.pth']

    data_root = '/DATA4_DB3/data/public/'

    # train_labeled_studies 和 test_labeled_studies 不需要，根据folder的名字来判断label
    train_image_paths = data_root + 'MURA-v1.1/train_image_paths.csv'   # 训练集存放路径
    # train_labeled_studies = '/DATA4_DB3/data/public/MURA-v1.1/train_labeled_studies.csv'
    test_image_paths = data_root + 'MURA-v1.1/valid_image_paths.csv'    # 测试集存放路径
    # test_labeled_studies = '/DATA4_DB3/data/public/MURA-v1.1/valid_labeled_studies.csv'

    output_csv_path = 'predictions.csv'

    # load_model_path = 'checkpoints/CustomDenseNet169_0613_14:42:38.pth'
    load_model_path = None                                        # 加载预训练的模型的路径，为None代表不加载

    batch_size = 8                                                  # batch size
    use_gpu = True                                                  # user GPU or not
    num_workers = 4                                                 # how many workers for loading data
    print_freq = 20                                                 # print info every N batch

    debug_file = 'tmp/debug'                                        # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 20
    lr = 0.0001                                                      # initial learning rate
    lr_decay = 0.5                                                  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-5                                             # 损失函数

    def parse(self, kwargs):
        """
        根据字典 kwargs 更新 config 参数。
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn(f'Warning: opt has no attribute {k}')
            setattr(self, k, v)

        print('User config: ')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))

    def __str__(self):
        print_dict = {
            "lr": self.lr,
            "lr_decay": self.lr_decay,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "time": time.strftime('%m%d_%H:%M:%S')
        }
        return "_".join(["{}:{}".format(k, v) for k, v in print_dict.items()])


opt = Config()
