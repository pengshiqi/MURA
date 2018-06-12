# -*- coding: utf-8 -*-

import warnings


class Config(object):
    env = 'MURA'                                                    # visdom 环境

    data_root = '/DATA4_DB3/data/public/'

    # train_labeled_studies 和 test_labeled_studies 不需要，根据folder的名字来判断label
    train_image_paths = '/DATA4_DB3/data/public/MURA-v1.1/train_image_paths.csv'   # 训练集存放路径
    # train_labeled_studies = '/DATA4_DB3/data/public/MURA-v1.1/train_labeled_studies.csv'
    test_image_paths = '/DATA4_DB3/data/public/MURA-v1.1/valid_image_paths.csv'    # 测试集存放路径
    # test_labeled_studies = '/DATA4_DB3/data/public/MURA-v1.1/valid_labeled_studies.csv'

    load_model_path = None                                          # 加载预训练的模型的路径，为None代表不加载

    batch_size = 8                                                  # batch size
    use_gpu = True                                                  # user GPU or not
    num_workers = 4                                                 # how many workers for loading data
    print_freq = 20                                                 # print info every N batch

    debug_file = 'tmp/debug'                                        # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 20
    lr = 0.0001                                                      # initial learning rate
    lr_decay = 0.3                                                  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4                                             # 损失函数

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
        }
        return "&".join(["{}:{}".format(k, v) for k, v in print_dict.items()])


opt = Config()
