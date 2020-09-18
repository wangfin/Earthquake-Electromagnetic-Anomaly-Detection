#!/usr/bin/env python
# @Time    : 2020/9/4 15:27
# @Author  : wb
# @File    : config.py

'''
配置文件
'''

import warnings


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    model = 'MSNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = './data/train/'  # 训练集存放路径
    test_data_root = './data/test1/'  # 测试集存放路径
    load_model_path = './checkpoints/MSNet_0918_12-52-01.pth'  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 128  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 1  # print info every N batch

    debug_file = './tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = './results/result.csv'

    max_epoch = 10
    lr = 0.1  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数


def parse(self, kwargs):
    '''
    根据字典kwargs 更新 config参数
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
# opt.parse = parse
