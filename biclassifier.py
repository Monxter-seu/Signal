import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
from torch.utils.data import DataLoader, TensorDataset


group_size=2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义训练数据和标签
#train_data = torch.randn(100, 64000)
#train_labels = torch.randint(0, 2, (100,))
train_data=np.ones((200,64000))
train_labels=torch.ones((200,))
train_labels=train_labels.to(device)
#print(train_data)
    
for i in range(100):
    train_data[i,:],_ = librosa.load('testDataTrue/newWave1-'+str(i+1)+'_s1.wav',sr=16000)

for i in range(100):
    train_data[i,:],_ = librosa.load('testDataFalse/newWave1-'+str(i+1)+'_s1.wav',sr=16000)

train_data=torch.from_numpy(train_data)
train_data=train_data.to(device)
train_labels[100:200]=torch.zeros((100,))

#print(train_labels)
#train_data=torch.tensor(train_data)
#train_labels=torch.randint(0, 2, (1,))
# 定义测试数据和标签
#test_data = torch.randn(20, 64000)
#test_labels = torch.randint(0, 2, (20,))

# 将数据和标签组成数据集
train_dataset = TensorDataset(train_data, train_labels)
#test_dataset = TensorDataset(test_data, test_labels)

# 定义数据加载器
batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
        self.fc3 = nn.Linear(256, 1)

        # 定义ReLU和Sigmoid激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 将输入张量展平
        #print('x.shape',x.shape)
        x = x.view(x.size(0), -1)
        #x=[B,T]
        #self.fc(x)
        #print('x.shape',x.shape)
        #mixture,_ = librosa.load('testDataTrue/newWave1-1_s1.wav',sr=16000)
        mixtures =x
        #print('mixture.type',type(mixture))
        #print('mixtures.shape',mixtures.shape)
        #分组，再进行fft
        #group_size = 2048
        #mixture=np.squeeze(mixture)
        spectrum = []
        for i in range(x.size(0)):
            num_elements = 31 * 2048
            mixture=mixtures[i,:]
            # 将张量裁剪为要保留的元素数量
            mixture = mixture[:num_elements]

            # 将张量重塑为形状为[31, 2048]的张量
            mixture_reshaped = mixture.reshape(31, 2048)

            # 对第二个维度执行FFT
            mixture_fft = torch.fft.fftn(mixture_reshaped, dim=1)

            # 计算形状为[31, 2048]的张量的平均值
            mixture_fft_mean = torch.mean(mixture_fft, dim=0)
            
            spectrum.append(mixture_fft_mean)
            
            #groups = [mixture[i:i+group_size] for i in range(0, len(mixture), group_size)]
            #print('groups[0].len',len(groups[0]))
            #print('groups.size',groups.size)
            #fft_groups = [np.fft.fft(group) for group in groups]
            #print('fft_groups[0].len',len(fft_groups[0]))
            #amp_spectrum = [np.abs(fft_group) for fft_group in fft_groups]
            #amp_spectrum.pop()
            
            #mean_amp_spectrum = np.mean(amp_spectrum,axis=0)
            #mean_amp_spectrum = np.reshape(mean_amp_spectrum,(1,2048))
            #spectrum.append(mean_amp_spectrum)
            #print(mean_amp_spectrum)
        #spectrum = np.array(spectrum)
        #x=torch.from_numpy(spectrum)
        x=torch.stack(spectrum,dim=0)
        #print('x.shape',x.shape)
        x=x.to(device)
        x=x.type(torch.FloatTensor)
        x=x.to(device)
        #print('x.size',x.size)
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

# 实例化模型
model = BinaryClassifier()
model = model.cuda()

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
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
    #print('train_loss',train_loss)

    # 测试模式
    model.eval()
    test_loss = 0
    test_correct = 0
    #with torch.no_grad():
     #   for data, labels in test_loader:
      #      # 将数据和标签转换为张量
       #     data = data.float()
            
torch.save(model.state_dict(), "biClassifier.pth")