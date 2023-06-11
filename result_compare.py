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
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.manifold import TSNE

TrueAddress='testData/True'
FalseAddress='testData/False'
# index=1
# i=1

group_size=2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义训练数据和标签
#train_data = torch.randn(100, 64000)
#train_labels = torch.randint(0, 2, (100,))
train_data=torch.ones((1600,2048))
test_data=torch.ones((400,2048))

train_labels=torch.ones((1600,))
test_labels=torch.ones((400,))

train_labels=train_labels.to(device)
test_labels=test_labels.to(device)

#target=[258,1096,699,694,1302,1156,1071,1754,1596,261,1340]
#target=[4,3,201,6,1856,1428,1473,455]
target=[201,1856,1428,1473,455]
#target=[201,1428]
target.sort()
#target=[258,1096,699]
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
            

fig=plt.figure()
#ax=plt.axes(projection='3d')
fig,ax=plt.subplots()

# for i in range(0,400):
    # tempList=[]
    # for index in target:
        # tempList.append(train_data[i][index])
    # ax.plot(tempList,color='red')
    
# for i in range(400,800):
    # tempList=[]
    # for index in target:
        # tempList.append(train_data[i][index])
    # ax.plot(tempList,color='blue')
tempTotal=[]
for i in range(0,800):
    tempList=[]
    for index in target:
        tempList.append(train_data[i][index])
    tempTotal.append(tempList)
    ax.scatter(tempList[0],tempList[1],color='red')
    
for i in range(800,1600):
    tempList=[]
    for index in target:
        tempList.append(train_data[i][index])
    tempTotal.append(tempList)
    ax.scatter(tempList[0],tempList[1],color='blue')
    
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(train_data)

for i in range(0,500):
    ax.scatter(X_tsne[i,0],X_tsne[i,1],color='red')
    
for i in range(500,800):
    ax.scatter(X_tsne[i,0],X_tsne[i,1],color='yellow')
    
for i in range(800,1300):  
    ax.scatter(X_tsne[i,0],X_tsne[i,1],color='green')
    
for i in range(1300,1600):  
    ax.scatter(X_tsne[i,0],X_tsne[i,1],color='blue')
    
# plt.show()
# fig,ax=plt.subplots()

# for i in range(0,1):
    # ax.plot(train_data[i],color='red')
    
# for i in range(400,402):
    # ax.plot(train_data[i],color='blue')
plt.title("2 Device Processes")

plt.show()