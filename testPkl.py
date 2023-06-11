import torch
import torch.nn as nn
import librosa
import pickle
from biclassifier import BinaryClassifier
from data import EvalDataLoader, EvalDataset
from conv_tasnet import ConvTasNet
from torch.utils.data import Dataset, DataLoader
import lime
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import csv

test_Data=np.ones((100,64000))
for i in range(100):
    test_Data[i,:],_ = librosa.load('testData/True/newWave4-'+str(i+1)+'.wav',sr=16000)
    
#test_sample=test_Data[2,:]

res=[]
with open('local_model.pkl', 'rb') as f:
   exp = pickle.load(f)
   for i in range(100):
        test_sample=test_Data[i,:]
        #local_pred = np.sum(exp.as_list()[0][1]*test_sample)+exp.intercept[1]
        local_pred = np.sum(exp.as_list()[0][1]*test_sample)
        res.append(local_pred)
        print(local_pred)
   f.close()
   
print('res_mean:',np.mean(res))

with open('TrueLime_true1.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(res)
    file.close()