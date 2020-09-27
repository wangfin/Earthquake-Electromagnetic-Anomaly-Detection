#!/usr/bin/env python
# @Time    : 2020/9/10 9:41
# @Author  : wb
# @File    : data_process.py

import numpy as np
import pandas as pd
from vmdpy import VMD
import random


def normalize(sample):
    """(0,1)normalization
    :param sample : the object which is a 1*576 vector to be normalized
    """
    sample_norm = (sample - min(sample)) / (max(sample) - min(sample))

    return sample_norm


def vmd(freq):
    K = 3
    alpha = 2000  # moderate bandwidth constraint数据保真度约束的平衡参数
    tau = 0.  # noise-tolerance (no strict fidelity enforcement)
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-7
    u, u_hat, omega = VMD(freq, alpha, tau, K, DC, init, tol)

    freq_VMD = u[0]
    for k in range(K - 1):
        freq_VMD = freq_VMD + u[k]
    return freq_VMD


def add_noise(signal):
    mu = 0
    sigma = 0.08
    for i in range(signal.size):
        signal[i] += random.gauss(mu, sigma)

    return signal


def cut_samples(org_signals, add_noise_or_not, vmd_or_not, normalize_or_not):
    """ get original signals to 6*200*576 samples, meanwhile normalize these samples
    :param org_signals :a 6*100076 matrix of 6 original signals
    """

    data_splitted = np.zeros(shape=(6, 200, 576))
    temporary_s = np.zeros(shape=(200, 576))

    for i in range(6):
        s = org_signals[i]
        for x in range(200):
            temporary_s[x] = s[500*x: 576+500*x]
            if add_noise_or_not:
                temporary_s[x] = add_noise(temporary_s[x])
            if vmd_or_not:
                temporary_s[x] = vmd(temporary_s[x])
            if normalize_or_not:
                temporary_s[x] = normalize(temporary_s[x])
        data_splitted[i] = temporary_s

    return data_splitted


def make_datasets(data):
    """ 输入切分后6*200*576的样本, 转换为1200*24*24，生成对应标签，并随机划分训练集和测试集。
     :param org_samples :(6, 200, 576) splitted data
    """
    # 合并为1200*756
    data_reshape = data.reshape((1200, 576))

    # 变形为1200*24*24
    data = data_reshape.reshape((1200, 24, 24))

    # 生成标签
    label = np.zeros(shape=(1200,), dtype=np.longlong)
    for i in range(6):
        label[200 * i: 200 * i + 200] = i

    # 数据划分
    train_index = random.sample(range(0, 1200), 1000)
    test_index = list(range(0, 1200, 1))
    test_index = list(set(test_index).difference(set(train_index)))
    random.shuffle(test_index)

    X_train = data[train_index]
    y_train = label[train_index]
    X_test = data[test_index]
    y_test = label[test_index]

    return X_train, y_train, X_test, y_test


def data_process(FileName, add_noise_or_not, vmd_or_not, normalize_or_not):
    data = pd.read_csv(FileName, header=None)
    data = cut_samples(data, add_noise_or_not, vmd_or_not, normalize_or_not)
    X_train, y_train, X_test, y_test = make_datasets(data)

    return X_train, y_train, X_test, y_test

