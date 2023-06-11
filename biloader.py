import torch
from torch.utils.data import Dataset, DataLoader

class BinaryDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input_ = torch.tensor(self.inputs[index], dtype=torch.float32)
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return input_, target

# 创建一个数据集
inputs = ... # [batch_size, num_features]的张量
targets = ... # [batch_size]的标签张量（0或1）

dataset = BinaryDataset(inputs, targets)

# 创建一个数据加载器
batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
