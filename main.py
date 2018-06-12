# -*- coding: utf-8 -*-

import os
import torch as t
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score#, confusion_matrix

from config import opt
from utils import Visualizer
from dataset import MURA_Dataset
from models import CustomDenseNet169


def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # step 1: configure model
    # model = densenet169(pretrained=True)
    model = CustomDenseNet169(num_classes=2)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    model.train()
    # step 2: data
    train_data = MURA_Dataset(opt.data_root, opt.train_image_paths, train=True)
    # val_data = MURA_Dataset(opt.data_root, opt.test_image_paths, train=False)
    val_data = MURA_Dataset(opt.data_root, opt.test_image_paths, test=True)

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # step 3: criterion and optimizer
    A = 21935
    N = 14873
    weight = t.Tensor([A/(A+N), N/(A+N)])
    if opt.use_gpu:
        weight = weight.cuda()
    criterion = t.nn.CrossEntropyLoss(weight=weight)
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)


    # step 4: meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # step 5: train
    s = t.nn.Softmax()
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label, _) in tqdm(enumerate(train_dataloader)):

            # train model
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.data[0])
            confusion_matrix.add(s(Variable(score.data)).data, target.data)

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])
                print('loss', loss_meter.value()[0])

                # debug
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        model.save()

        # validate and visualize
        val_cm, val_accuracy, val_loss = val(model, val_dataloader)

        vis.plot('val_accuracy', val_accuracy)
        print('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))
        print("epoch:{epoch},lr:{lr},loss:{loss},val_loss:{val_loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_loss=val_loss, val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))


        # update learning rate
        # if loss_meter.value()[0] > previous_loss:
        if val_loss > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = val_loss


def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    s = t.nn.Softmax()

    criterion = t.nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()

    for ii, data in tqdm(enumerate(dataloader)):
        input, label, _ = data
        val_input = Variable(input, volatile=True)
        target = Variable(label)
        if opt.use_gpu:
            val_input = val_input.cuda()
            target = target.cuda()
        score = model(val_input)
        # confusion_matrix.add(softmax(score.data.squeeze()), label.type(t.LongTensor))
        confusion_matrix.add(s(Variable(score.data.squeeze())).data, label.type(t.LongTensor))
        loss = criterion(score, target)
        loss_meter.add(loss.data[0])

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    loss = loss_meter.value()[0]

    return confusion_matrix, accuracy, loss


def test(**kwargs):
    opt.parse(kwargs)

    # configure model
    model = CustomDenseNet169(num_classes=2)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    model.eval()
    # data
    test_data = MURA_Dataset(opt.data_root, opt.test_image_paths, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    results = []
    confusion_matrix = meter.ConfusionMeter(2)
    s = t.nn.Softmax()

    for ii, (data, label, path) in tqdm(enumerate(test_dataloader)):
        input = Variable(data, volatile=True)
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)

        confusion_matrix.add(s(Variable(score.data.squeeze())).data, label.type(t.LongTensor))

        probability = t.nn.functional.softmax(score)[:, 0].data.tolist()
        # label = score.max(dim = 1)[1].data.tolist()

        # 每一行为 图片路径 和 positive的概率
        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]

        results += batch_results

    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())

    print('confusion matrix: ')
    print(cm_value)
    print(f'accuracy: {accuracy}')

    write_csv(results, opt.result_file)

    calculate_cohen_kappa()
    # return results


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'probability'])
        writer.writerows(results)


def calculate_cohen_kappa(threshold=0.5):
    input_csv_file_path = 'result.csv'

    result_dict = {}
    with open(input_csv_file_path, 'r') as F:
        d = F.readlines()[1:]
        for data in d:
            (path, prob) = data.split(',')

            folder_path = path[:path.rfind('/')]
            prob = float(prob)

            if folder_path in result_dict.keys():
                result_dict[folder_path].append(prob)
            else:
                result_dict[folder_path] = [prob]

    for k, v in result_dict.items():
        result_dict[k] = np.mean(v)
        # visualize
        # print(k, result_dict[k])

    XR_type_list = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']

    for XR_type in XR_type_list:

        # 提取出 XR_type 下的所有folder路径，即 result_dict 中的key
        keys = [k for k, v in result_dict.items() if k.split('/')[6] == XR_type]

        y_true = [1 if key.split('_')[-1] == 'positive' else 0 for key in keys]
        y_pred = [0 if result_dict[key] >= threshold else 1 for key in keys]

        print('--------------------------------------------')
        # print(XR_type)
        # print(y_true[:20])
        # print(y_pred[:20])

        kappa_score = cohen_kappa_score(y_true, y_pred)

        print(XR_type, kappa_score)

        # 预测准确的个数
        count = 0
        for i in range(len(y_true)):
            if y_pred[i] == y_true[i]:
                count += 1
        print(XR_type, 'Accuracy', 100.0 * count / len(y_true))


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
