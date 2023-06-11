#!/usr/bin/env python
# Created on 2023/05
# Author: HUA

import torch
import torch.nn as nn
import librosa
import pickle
from data import EvalDataLoader, EvalDataset
from conv_tasnet import ConvTasNet
from torch.utils.data import Dataset, DataLoader
import lime
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from testCSV import BinaryClassifier
from collections import Counter
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = BinaryClassifier()
net.load_state_dict(torch.load("speClassifier.pth"))
net.to(device)

def find_top_20_numbers(lst):
    counter = Counter(lst)
    top_20 = counter.most_common(20)
    return top_20
    
    
def batch_predict(data, model=net):
    """
    model: pytorch训练的模型, **这里需要有默认的模型**
    data: 需要预测的数据
    """
    data=data.reshape((-1,2048))
    #print('data.shape',data.shape)
    sample_num=data.shape[0]
    logit=np.array([])
    #print('size=',sample_num)
    #print('data.shape',data.shape)
    with torch.no_grad():
        for i in range(0,sample_num):
            temp_data=data[i]
            temp_data=temp_data.reshape((-1,2048))
            X_tensor = torch.from_numpy(temp_data).float()
            X_tensor = X_tensor.cuda()
        #print('++++++++++++X_tensor.type',type(X_tensor))
        #model.eval()
        #model.to(device)
        #with torch.no_grad():
            logits = model(X_tensor.cuda())
            logits = logits.detach().cpu().numpy()
            #print('logits=======',logits)
            logits=np.hstack((logits[0],1-logits[0]))
            #new_col = np.array([1-logits[0]])
            #print('new_col:',new_col)
            #logits = np.insert(logits, 1, new_col, axis=1)
            #print('final_logits',logits)
            logit=np.hstack((logit,logits))
        #_, predicted = torch.max(logits, 1)
    logit=logit.reshape(-1,2)
    # print('logit==',logit)
    #logits
    return logit

net.eval()

test_Data=np.ones((100,2048))
index=5

TrueAddress='testData/True'
FalseAddress='testData/False'
for i in range(1,101):
        with open(TrueAddress+'/newWave'+str(index)+'-'+str(i)+'.csv', 'r') as file:
            reader = csv.reader(file)
            row = next(reader)
            test_Data[i-1,0:2048] = (np.array(row, dtype=np.float32))
            file.close()

occurence = []
print('occurence_type',type(occurence))

for i in range(0,100):

    test_sample=test_Data[i,:]
    #print(test_sample.shape)

    explainer = LimeTabularExplainer(test_Data, mode="classification")

    #explainer = LimeTabularExplainer(test_Data, mode="classification", feature_names=feature_id, class_names=["0","1"])
    exp = explainer.explain_instance(test_sample,batch_predict, num_features=30, num_samples=50)

    #print(type(exp))


    # 提取特征的重要性得分（权重）
    weights = exp.local_exp[1]  # 1代表模型预测的标签索引
    #print(weights)
    # 提取特征索引和对应的权重得分
    feature_indices = [item[0] for item in weights]
    occurence.extend(feature_indices)
    feature_scores = [item[1] for item in weights]

    # 将特征索引和对应的得分导出到列表或NumPy数组
    feature_importance = list(zip(feature_indices, feature_scores))


result = find_top_20_numbers(occurence)
print(result)

#print(feature_importance)

# for i in exp.as_list():
        # # f.write(i)
    # print(i)