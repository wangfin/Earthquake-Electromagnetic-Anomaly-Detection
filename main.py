#!/usr/bin/env python
# @Time    : 2020/9/4 15:28
# @Author  : wb
# @File    : main.py

# 主文件

from config import opt
import os
import torch as t
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm
import torch


def test(**kwargs):
    opt.parse(kwargs)

    # 是否使用GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    # 是否使用GPU
    if opt.use_gpu: model.to(DEVICE)

    # data
    train_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (data, label) in enumerate(test_dataloader):
        # input = t.autograd.Variable(data, volatile=True)
        #if opt.use_gpu: input = input.to(DEVICE)

        data = data.to(DEVICE)
        label = label.to(DEVICE)
        score = model(data)
        probability = t.nn.functional.softmax(score)[:, 0].data.tolist()
        # label = score.max(dim = 1)[1].data.tolist()

        batch_results = [(path_, probability_) for path_, probability_ in zip(label, probability)]

        results += batch_results
    write_csv(results, opt.result_file)

    return results


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # 是否使用GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.to(DEVICE)

    # step2: data
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=False, num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # step4: meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # train
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader), total=len(train_data)):

            # train model
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            score = model(data)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.item())
            confusion_matrix.add(score.data, label.data)

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])

        model.save()

        # validate and visualize
        val_cm, val_accuracy = val(model, val_dataloader)

        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


def val(model, dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.type(t.LongTensor), volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def help():
    '''
    打印帮助的信息： python file.py help
    '''

    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire

    fire.Fire()


