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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 假设已经训练好了两个网络模块 net1 和 net2
net1 = ConvTasNet.load_model('final.pth.tar')
net1.to(device)
net2 = BinaryClassifier()
net2.load_state_dict(torch.load("biclassifier.pth"))

net2.to(device)
# 自定义一个网络结构，将 net1 的输出结果乘以 2，并将乘法结果输入到 net2 中
class Net(nn.Module):
    def __init__(self, net1, net2):
        super(Net, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        print('----x--- ', type(x), x)
        x1 = self.net1(x)
        x2 = torch.squeeze(x1, dim=1)
        #print('2222')
        out = self.net2(x2)
        return out

# 将 net1 和 net2 连接起来，形成一个新的网络
net = Net(net1, net2)

net.eval()

def batch_predict(data, model=net):
    """
    model: pytorch训练的模型, **这里需要有默认的模型**
    data: 需要预测的数据
    """
    data=data.reshape((-1,64000))
    #print('data.shape',data.shape)
    sample_num=data.shape[0]
    logit=np.array([])
    #print('size=',sample_num)
    #print('data.shape',data.shape)
    with torch.no_grad():
        for i in range(0,sample_num):
            temp_data=data[i]
            temp_data=temp_data.reshape((-1,64000))
            X_tensor = torch.from_numpy(temp_data).float()
            X_tensor = X_tensor.cuda()
        #print('++++++++++++X_tensor.type',type(X_tensor))
        #model.eval()
        #model.to(device)
        #with torch.no_grad():
            logits = model(X_tensor.cuda())
            logits = logits.detach().cpu().numpy()
            print('logits=======',logits)
            logits=np.hstack((logits[0],1-logits[0]))
            #new_col = np.array([1-logits[0]])
            #print('new_col:',new_col)
            #logits = np.insert(logits, 1, new_col, axis=1)
            #print('final_logits',logits)
            logit=np.hstack((logit,logits))
        #_, predicted = torch.max(logits, 1)
    logit=logit.reshape(-1,2)
    #print(logit)
    #logits
    return logit
    #return predicted.detach().cpu().numpy()
    #return logits.detach().cpu().numpy()
    #probs = torch.nn.functional.softmax(logits, dim=1)
    #probs.to(device)
    #return probs.detach().cpu().numpy()
    
#test_Data =  np.random.rand(4, 64000)
#test_sample=test_Data(1,:)
#prob = batch_predict(data=test_Data, model=net)
#print('prob_____',prob)    


test_Data=np.ones((10,64000))

#train_labels=torch.ones((20,))
#train_labels=train_labels.to(device)
#print(train_data)
    
for i in range(10):
    test_Data[i,:],_ = librosa.load('testData/True/newWave5-'+str(i+1)+'.wav',sr=16000)
    
test_sample=test_Data[6,:]
print(test_sample.shape)
#for i in range(1,11):
#    train_data[i+9,:],_ = librosa.load('testDataFalse/newWave1-'+str(i)+'_s1.wav',sr=16000)
    
    
num_arrays = 64000
array_length = 100
feature_id=[]

 #循环创建数组
for i in range(num_arrays):
    feature_id.append('feature_' + str(i+1));
    
    
#prob = batch_predict(data=test_sample, model=net)
#print(prob)    

#print(feature_id.size())


explainer = LimeTabularExplainer(test_Data, mode="classification")


#explainer = LimeTabularExplainer(test_Data, mode="classification", feature_names=feature_id, class_names=["0","1"])


exp = explainer.explain_instance(test_sample,batch_predict, num_features=30, num_samples=200)
#exp = explainer.explain_instance(test_sample,batch_predict,num_samples=2)

#使用本地模型进行预测
#print('exp.intercept=======',exp.intercept[0])
local_pred = np.sum(exp.as_list()[0][1]*test_sample)+exp.intercept[1]
#local_pred = np.sum(exp.as_list()[0][1]*test_sample)
print('Local model prediction:', local_pred)


#输出截距
#print('type of intercept is',type(exp.intercept))
#print('Output intercept:',exp.intercept.items())
#print('Output intercept:',exp.intercept)

#for i in exp.intercept:
#    print(i)
#输出解释
print('Explanation for instance:')

# with open('output.txt', 'w') as f:
   
for i in exp.as_list():
        # f.write(i)
    print(i)
    # f.close()

#保存本地模型
with open('local_model.pkl', 'wb') as f:
   pickle.dump(exp, f)
   f.close()

# 对新的网络进行测试
#with torch.no_grad():
#    test_data = torch.randn(2, 64000)
#    test_labels = torch.randint(0, 2, (2,))
#    test_data.to(device)
#    output = net(test_data)
    #explainer = LimeTabularExplainer(test_data, mode="classification")
    #explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns.tolist(), class_names=['0', '1'], discretize_continuous=True)
#    print(output)