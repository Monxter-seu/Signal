import torch
from scipy import signal

def lowpass_filter(data, cutoff_freq, fs):
    data=data.numpy()
    b, a = signal.butter(4, cutoff_freq / (fs / 2), 'lowpass', analog=False)
    smoothed = signal.filtfilt(b, a, data)
    smoothed=smoothed.copy()
    smoothed=torch.from_numpy(smoothed)
    return smoothed

# def lowpass_filter(input_data, cutoff_freq, sampling_freq, order=5):
    # nyquist_freq = 0.5 * sampling_freq
    # normalized_cutoff_freq = cutoff_freq / nyquist_freq
    # b, a = signal.butter(order, normalized_cutoff_freq, btype='low', analog=False)

    # input_shape = input_data.size()
    # num_samples = input_shape[0]

    # # 将滤波器系数转换为PyTorch张量
    # b = torch.tensor(b, dtype=torch.float32)
    # a = torch.tensor(a, dtype=torch.float32)

    # # 将滤波器系数转换为[1, 1, filter_order + 1]的形状
    # b = b.view(1, 1, -1)
    # a = a[1:].view(1, 1, -1)

    # # 在输入数据前后填充零
    # padding = torch.zeros(b.size(2) - 1, dtype=input_data.dtype)
    # padded_input_data = torch.cat([padding, input_data, padding])

    # # 将输入数据的形状调整为[1, 1, num_samples + 2 * padding_length]
    # padded_input_data = padded_input_data.view(1, 1, num_samples + 2 * (b.size(2) - 1))

    # # 应用滤波器
    # filtered_data = torch.nn.functional.conv1d(padded_input_data, b)
    # filtered_data = torch.nn.functional.conv1d(filtered_data, a)

    # # 裁剪输出数据为原始长度
    # filtered_data = filtered_data[..., b.size(2) - 1:-(b.size(2) - 1)].squeeze()

    return filtered_data

# # 示例使用
# # 创建随机输入张量
# input_data = torch.randn(1000)  # 输入形状为[1000]

# # 低通滤波器参数
# cutoff_frequency = 10.0  # 截止频率
# sampling_frequency = 100.0  # 采样频率
# filter_order = 4  # 滤波器阶数

# # 应用低通滤波器
# filtered_data = lowpass_filter(input_data, cutoff_frequency, sampling_frequency, filter_order)

# # 输出结果
# print(filtered_data)