#!/usr/bin/env python
# @Time    : 2020/9/4 15:28
# @Author  : wb
# @File    : main.py

# 主文件

from config import opt
import os
import torch as t
import models
from data.data_process import data_process
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm
import torch
import copy
import random
import numpy as np
from torch import nn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def train(**kwargs):
    setup_seed(20)
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # 是否使用GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.to(DEVICE)

    # step2: data
    X_train, y_train, X_test, y_test = data_process(opt.data_root, add_noise_or_not=False, vmd_or_not=True, normalize_or_not=True)
    X_train, y_train, X_test, y_test = torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test)
    X_train = torch.unsqueeze(X_train, dim=1).type(torch.FloatTensor)
    X_test = torch.unsqueeze(X_test, dim=1).type(torch.FloatTensor)

    train_data = Data.TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    val_data = Data.TensorDataset(X_test, y_test)
    val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # step4: meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(6)
    previous_loss = 1e100

    # 保存最优模型
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # train
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_loader), total=len(train_data)//opt.batch_size):

            # train model
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.item())
            predicton = torch.argmax(outputs.data, dim=1)
            confusion_matrix.add(predicton, label.data)

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])
                print('t = %d, loss = %.4f' % (ii + 1, loss_meter.value()[0]))

        # validate and visualize
        val_cm, val_accuracy = val(model, val_loader)

        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))

        # save the best model
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]

    # 加载最佳的参数
    model.load_state_dict(best_model_wts)

    # 保存模型
    model_save_path = model.save()
    print('最优模型保存在：', model_save_path)


def val(model, dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    # 是否使用GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    confusion_matrix = meter.ConfusionMeter(6)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input)
        val_label = Variable(label.type(t.LongTensor))
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        outputs = model(val_input)
        predicton = torch.argmax(outputs.data, dim=1)
        confusion_matrix.add(predicton, label)

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1] + cm_value[2][2] + cm_value[3][3] + cm_value[4][4] + cm_value[5][5]) / (cm_value.sum())
    return confusion_matrix, accuracy


def incremental_train(**kwargs):
    setup_seed(20)
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # step1: configure model
    # 是否使用GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 导入teacer模型
    teacher_model = getattr(models, opt.model)()
    if opt.load_model_path:
        teacher_model.load(opt.load_model_path)
    if opt.use_gpu:
        teacher_model.to(DEVICE)

    # 配置student模型
    student_model = getattr(models, opt.model)()
    if opt.use_gpu:
        student_model.to(DEVICE)

    # step2: data
    # X_train中生成100加噪+X_test中100不加噪训练，同样再做200分别用来测试
    X_train, y_train, X_test, y_test = data_process(opt.data_root, add_noise_or_not=False, vmd_or_not=True, normalize_or_not=True)
    X_train_n, y_train_n, X_test_n, y_test_n = data_process(opt.data_root, add_noise_or_not=True, vmd_or_not=True, normalize_or_not=True)
    # 增量训练集
    X_incremental_train = np.vstack((X_train_n[0:100, :, :], X_test[0:100, :, :]))
    y_incremental_train = np.hstack((y_train_n[0:100], y_test[0:100]))
    # 增量测试集
    X_incremental_test = np.vstack((X_train_n[100:200, :, :], X_test[100:200, :, :]))
    y_incremental_test = np.hstack((y_train_n[100:200], y_test[100:200]))

    X_train, y_train, X_test, y_test = torch.tensor(X_incremental_train), torch.tensor(y_incremental_train), torch.tensor(X_incremental_test), torch.tensor(y_incremental_test)
    X_train = torch.unsqueeze(X_train, dim=1).type(torch.FloatTensor)
    X_test = torch.unsqueeze(X_test, dim=1).type(torch.FloatTensor)

    train_data = Data.TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    val_data = Data.TensorDataset(X_test, y_test)
    val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    criterion2 = nn.KLDivLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(student_model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # step4: meters
    loss_meter = meter.AverageValueMeter()
    loss1_meter = meter.AverageValueMeter()
    loss2_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(6)
    previous_loss = 1e100

    # 保存最优模型
    best_model_wts = copy.deepcopy(student_model.state_dict())
    best_acc = 0.0

    # train
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_loader), total=len(train_data)//opt.batch_size):

            # train model
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            soft_target = teacher_model(data)

            optimizer.zero_grad()
            outputs = student_model(data)

            loss1 = criterion(outputs, label)
            T = 2
            outputs_S = nn.functional.log_softmax(outputs / T, dim=1)
            outputs_T = nn.functional.softmax(soft_target / T, dim=1)

            loss2 = criterion2(outputs_S, outputs_T) * T * T

            loss = loss1 * (1 - opt.alpha) + loss2 * opt.alpha

            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.item())
            predicton = torch.argmax(outputs.data, dim=1)
            confusion_matrix.add(predicton, label.data)
            print('[%d, %5d] loss: %.4f loss1: %.4f loss2: %.4f' % (epoch + 1, (ii + 1) * opt.batch_size, loss.item(), loss1.item(), loss2.item()))

        # validate and visualize
        val_cm, val_accuracy = val(student_model, val_loader)

        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))

        # save the best model
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model_wts = copy.deepcopy(student_model.state_dict())

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]

    # 加载最佳的参数
    student_model.load_state_dict(best_model_wts)

    # 保存模型
    model_save_path = student_model.save()
    print('最优模型保存在：', model_save_path)


def test(**kwargs):
    setup_seed(20)
    opt.parse(kwargs)

    # 是否使用GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    # 是否使用GPU
    if opt.use_gpu:
        model.to(DEVICE)

    # 数据读取
    X_train, y_train, X_test, y_test = data_process(opt.data_root, add_noise_or_not=False, vmd_or_not=True, normalize_or_not=True)
    X_train_n, y_train_n, X_test_n, y_test_n = data_process(opt.data_root, add_noise_or_not=True, vmd_or_not=True, normalize_or_not=True)
    # 无噪声测试集
    X_test = X_test[100:150, :, :]
    y_test = y_test[100:150]
    # 有噪声测试集
    X_test_n = X_test_n[150:200, :, :]
    y_test_n = y_test_n[150:200]

    # 变成torch.Tensor
    X_test, y_test, X_test_n, y_test_n = torch.tensor(X_test), torch.tensor(y_test), torch.tensor(X_test_n), torch.tensor(y_test_n)

    # (1000, 24, 24)变形为(1000, 1, 24, 24)
    X_test = torch.unsqueeze(X_test, dim=1).type(torch.FloatTensor)
    X_test_n = torch.unsqueeze(X_test_n, dim=1).type(torch.FloatTensor)

    # 数据装入DataLoader
    test_data = Data.TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_data_n = Data.TensorDataset(X_test_n, y_test_n)
    test_loader_n = DataLoader(dataset=test_data_n, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # 保存输出结果
    results = []
    confusion_matrix = meter.ConfusionMeter(6)
    for ii, (data, label) in enumerate(test_loader):
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        outputs = model(data)
        outputs = nn.functional.softmax(outputs)
        predicton = torch.argmax(outputs.data, dim=1)
        confusion_matrix.add(predicton, label)
        probability = outputs.data.tolist()

        batch_results = [(path_, probability_) for path_, probability_ in zip(label, probability)]
        results += batch_results
    write_csv(results, opt.result_file)
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1] + cm_value[2][2] + cm_value[3][3] + cm_value[4][4] + cm_value[5][5]) / (cm_value.sum())

    confusion_matrix = meter.ConfusionMeter(6)
    for ii, (data, label) in enumerate(test_loader_n):
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        outputs = model(data)
        outputs = nn.functional.softmax(outputs)
        predicton = torch.argmax(outputs.data, dim=1)
        confusion_matrix.add(predicton, label)

    cm_value = confusion_matrix.value()
    accuracy_n = 100. * (cm_value[0][0] + cm_value[1][1] + cm_value[2][2] + cm_value[3][3] + cm_value[4][4] + cm_value[5][5]) / (cm_value.sum())

    print("test acc = %5d%%, test_n acc = %5d%%" % (accuracy, accuracy_n))

    return results


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
    # train()
    # incremental_train()
    test()
