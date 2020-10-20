#!/usr/bin/env python
# @Time    : 2020/9/10 9:41
# @Author  : wb
# @File    : data_process.py

# 数据读取处理

import struct
import os

filename = './wenchang/52000001_0001_20190422_164932_20MHz_6000MHz_27.343kHz_V_M.bin'

binfile = open(filename, 'rb') # 打开二进制文件
size = os.path.getsize(filename) # 获得文件大小
# for i in range(size):
#     data = binfile.read(1) #每次输出一个字节
#     print(data)
# binfile.close()

for i in range(size):
    data = binfile.read(1)
    num = struct.unpack('B', data)
    print(num)







