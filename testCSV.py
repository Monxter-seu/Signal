#!/usr/bin/env python
# Created on 2023/03
# Author: HUA

import torch
import torch.nn as nn
import os
import numpy as np
import librosa
import torch.optim as optim
import csv
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader, TensorDataset

group_size=2048

# 定义模型
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()

        # 定义三个全连接层
        #self.fc1 = nn.Linear(2048, 256)
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 256)
        #self.fc2 = nn.Linear(256, 256)
        #self.fc3 = nn.Linear(256, 1)
        #self.fc2_1 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        # 定义ReLU和Sigmoid激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 应用第一个全连接层和ReLU激活函数
        x = self.fc1(x)
        x = self.relu(x)

        # 应用第二个全连接层和ReLU激活函数
        x = self.fc2(x)
        x = self.relu(x)

        # 应用第三个全连接层和Sigmoid激活函数
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x

# print(tensor_read)
# print(tensor_read.shape)

if __name__ == "__main__":
    testTrueAddress='vData/True'
    testFalseAddress='vData/False'
    trainTrueAddress='testData/True'
    trainFalseAddress='testData/False'
    
    #读取True数据与False训练集数据
    trueDir=os.listdir(trainTrueAddress)
    falseDir=os.listdir(trainFalseAddress)
    trueDevicesNums=len(trueDir)
    falseDevicesNums=len(falseDir)
    
    totalTrue=0
    totalFalse=0
    
    for sonDir in trueDir:
        sonsonDir=os.listdir(sonDir)
        totalTrue+=len(sonsonDir)
        
    for sonDir in falseDir:
        sonsonDir=os.listdir(sonDir)
        totalFalse+=len(sonsonDir)
        
        
    #读取True数据与False测试集数据
    trueDir=os.listdir(testTrueAddress)
    falseDir=os.listdir(testFalseAddress)
    trueDevicesNums=len(trueDir)
    falseDevicesNums=len(falseDir)
    

    

    # index=1
    # i=1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 定义训练数据和标签
    #train_data = torch.randn(100, 64000)
    #train_labels = torch.randint(0, 2, (100,))
    train_data=torch.ones((totalTrue+totalFalse,group_size))
    test_data=torch.ones((400,group_size))

    train_labels=torch.ones((1600,))
    test_labels=torch.ones((400,))

    train_labels=train_labels.to(device)
    test_labels=test_labels.to(device)

    #print(train_data)
    for index in range(1,9):
        for i in range(1,101):
            with open(TrueAddress+'/newWave'+str(index)+'-'+str(i)+'.csv', 'r') as file:
                reader = csv.reader(file)
                row = next(reader)
                train_data[(index-1)*100+i-1,0:2048] = torch.from_numpy(np.array(row, dtype=np.float32))
                file.close()
            with open(FalseAddress+'/newWave'+str(index)+'-'+str(i)+'.csv', 'r') as file:
                reader = csv.reader(file)
                row = next(reader)
                train_data[(index+7)*100+i-1,0:2048] = torch.from_numpy(np.array(row, dtype=np.float32))
                file.close()


    for index in range(9,11):
        for i in range(1,101):
            with open(TrueAddress+'/newWave'+str(index)+'-'+str(i)+'.csv', 'r') as file:
                reader = csv.reader(file)
                row = next(reader)
                test_data[i-1+100*(index-9),0:2048] = torch.from_numpy(np.array(row, dtype=np.float32))
                file.close()
            with open(FalseAddress+'/newWave'+str(index)+'-'+str(i)+'.csv', 'r') as file:
                reader = csv.reader(file)
                row = next(reader)
                test_data[100*(index-9)+i-1+200,0:2048] = torch.from_numpy(np.array(row, dtype=np.float32))
                file.close()
                
    #train_data=torch.from_numpy(train_data)
    train_data=train_data.to(device)
    test_data=test_data.to(device)

    train_labels[800:1600]=torch.zeros((800,))
    test_labels[200:400]=torch.zeros((200,))

    #print(train_labels)
    #train_data=torch.tensor(train_data)
    #train_labels=torch.randint(0, 2, (1,))
    # 定义测试数据和标签
    #test_data = torch.randn(20, 64000)
    #test_labels = torch.randint(0, 2, (20,))

    # 将数据和标签组成数据集
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # 定义数据加载器
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # with open(address+'/newWave'+str(index)+'-'+str(i)+'.csv', 'r') as file:
        # reader = csv.reader(file)
        # row = next(reader)
        # tensor_read = torch.from_numpy(np.array(row, dtype=np.float32))
    # 实例化模型
    model = BinaryClassifier()
    model = model.cuda()

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # 训练模型
    num_epochs = 40
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0
        train_correct = 0
        #print('train_loader.shape',train_loader.shape)
        for data, labels in train_loader:
            # 将数据和标签转换为张量
            data = data.float()
            labels = labels.float()

            # 向前传递
            outputs = model(data)
            loss = criterion(outputs, labels.unsqueeze(1))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算损失和准确率
            train_loss += loss.item() * data.size(0)
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels.unsqueeze(1)).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)
        print('train_loss',train_loss)
        print('train_accuracy',train_accuracy)

    # 测试模式
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
       for data, labels in test_loader:
           # 将数据和标签转换为张量
            data = data.float()
            labels = labels.float()

            # 向前传递
            outputs = model(data)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            test_loss += loss.item() * data.size(0)
            predicted = (outputs > 0.5).float()
            test_correct += (predicted == labels.unsqueeze(1)).sum().item()
            test_loss /= len(test_loader.dataset)
            test_accuracy = test_correct / len(test_loader.dataset)
    print('test_accuracy',test_accuracy)
    torch.save(model.state_dict(), "speClassifier.pth")
    print('222222222222')