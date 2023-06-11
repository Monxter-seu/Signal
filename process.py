#!/usr/bin/env python
# Created on 2023/03
# Author: HUA

import torch
import torch.nn as nn
import numpy as np
import faiss

#进行去掉padding，并且降维的操作，之后进行分组fft，最后将网络的结果输入KNN分类器
class ProcessingLayer(nn.Module):
    def __init__(self):
        super(PreprocessingLayer, self).__init__()
        #self.mean = torch.tensor(mean).view(1, -1, 1, 1) # 输入的均值，用于标准化
        #self.std = torch.tensor(std).view(1, -1, 1, 1) # 输入的标准差，用于标准化

    def forward(self, x):
        # [B,C,T]是输入的数据，!!!  mix_lengths不知道如何确定
        estimate_source = model(x)
        flat_estimate = remove_pad(estimate_source, mix_lengths)
        #mixture长度应该是64000
        mixture = remove_pad(mixture, mix_lengths)
        #分组，再进行fft
        group_size = 2048
        groups = [input_data[i:i+group_size] for i in range(0, len(mixture), group_size)]
        fft_groups = [np.fft.fft(group) for group in groups]
        amp_spectrum = [np.abs(fft_group) for fft_group in fft_groups]
        mean_amp_spectrum = np.mean(amp_spectrum, axis=0)
        
        
        return x
        

class KNNClassifier(nn.Module):
    def __init__(self, n_neighbors=10):
        super(KNNClassifier, self).__init__()
        self.n_neighbors = n_neighbors

    def forward(self, x, train_features, train_labels):
        # 构建索引
        index = faiss.IndexFlatL2(train_features.shape[1])
        index.add(train_features.numpy())

        # 进行KNN搜索
        distances, indices = index.search(x.numpy(), self.n_neighbors)

        # 计算K个邻居的标签的众数
        indices_tensor = torch.from_numpy(indices)
        k_labels = torch.gather(train_labels, dim=0, index=indices_tensor)
        pred_labels, _ = torch.mode(k_labels, dim=1)

        return pred_labels

if __name__ == "__main__":
    #parser = argparse.ArgumentParser("WSJ0 data preprocessing")
    #parser.add_argument('--in-dir', type=str, default=None,
    #                    help='Directory path of wsj0 including tr, cv and tt')
    #parser.add_argument('--out-dir', type=str, default=None,
    #                    help='Directory path to put output files')
    #parser.add_argument('--sample-rate', type=int, default=8000,
    #                    help='Sample rate of audio file')
    #args = parser.parse_args()
    #print(args)
    #preprocess(args)
