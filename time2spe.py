#!/usr/bin/env python
# Created on 2023/03
# Author: HUA

import torch
import torch.nn as nn
import numpy as np
import librosa
import csv
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
from scipy import signal
from lowpass import lowpass_filter


arguments = sys.argv

if(arguments[1]=='TRUE'):
    address='testData/True'
else:
    address='testData/False'
    
for index in range(1,11):

    for i in range(1,101):

        oname=address+'/newWave'+str(index)+'-'+str(i)+'.wav'
        mixture,_ = librosa.load(oname,sr=16000)
        mixture=torch.from_numpy(mixture)
        #print('mixture.type',type(mixture))
        #print('mixture.shape',mixture.shape)

        #分组，再进行fft
        #group_size = 2048
        #groups = [mixture[i:i+group_size] for i in range(0, len(mixture), group_size)]
        #print('groups.len',len(groups[0]))
        #print('groups.size',groups.size)
        #fft_groups = [np.fft.fft(group) for group in groups]
        #amp_spectrum = [np.abs(fft_group) for fft_group in fft_groups]
        #amp_spectrum.pop()
        num_elements = 31 * 2048

        # 将张量裁剪为要保留的元素数量
        mixture = mixture[:num_elements]

        # 将张量重塑为形状为[31, 2048]的张量
        mixture_reshaped = mixture.reshape(31, 2048)


        # 对第二个维度执行FFT
        mixture_fft = torch.fft.fftn(mixture_reshaped, dim=1)
        mixture_fft=torch.abs(mixture_fft)
        # 计算形状为[32, 2048]的张量的平均值
        mixture_fft_mean = torch.mean(mixture_fft, dim=0)




        # # 定义平滑窗口大小和截止频率
        # window_size = 1
        # cutoff_freq = 10

        # # 创建低通滤波器
        # cutoff = int(window_size * cutoff_freq)
        # lowpass_filter = torch.tensor([1.0] * cutoff + [0.0] * (window_size - cutoff), dtype=torch.float32)

        # # 归一化滤波器
        # lowpass_filter = lowpass_filter / lowpass_filter.sum()

        # # 使用低通滤波器平滑处理张量
        # smoothed_tensor = F.conv1d(mixture_fft_mean.unsqueeze(0).unsqueeze(0), lowpass_filter.unsqueeze(0).unsqueeze(0), padding=window_size//2).squeeze()
        # mixture_fft=smoothed_tensor
        # 绘制原始张量和平滑后的张量
        
        smoothed_tensor=lowpass_filter(mixture_fft_mean,20,2048)
        
        # plt.figure(figsize=(12, 6))
        #plt.plot(mixture_fft_mean, label='Original')
        plt.plot(smoothed_tensor, label='Smoothed')
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('Smoothed Tensor')
        # plt.legend()
        # plt.show()

        with open(address+'/newWave'+str(index)+'-'+str(i)+'.csv', 'w+') as file:
            writer = csv.writer(file)
            writer.writerow(smoothed_tensor.numpy())
            file.close()
            
plt.show()