import os
import wave
import numpy as np
import scipy.io

def write_wav_files(data_folder, output_folder, segment_length, wav_length, sample_rate):
    # 检查输出文件夹是否存在，如果不存在则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取数据文件夹中的所有.mat文件
    mat_files = [file for file in os.listdir(data_folder) if file.endswith('.mat')]

    for file in mat_files:
        # 读取.mat文件
        mat_data = scipy.io.loadmat(os.path.join(data_folder, file))
        
        # 获取数据
        data = mat_data['rawData12']
        
        # 计算数据段数
        num_segments = len(data) // segment_length

        # 创建WAV文件并写入数据
        for i in range(num_segments):
            segment = data[i*segment_length:(i+1)*segment_length, :]
            
            # 将数据按行拼接为单声道音频
            audio_data = np.ravel(segment, order='F')
            
            # 创建WAV文件路径和文件名
            output_file = os.path.splitext(file)[0] + f'_segment{i+1}.wav'
            output_path = os.path.join(output_folder, output_file)
            
            # 将数据归一化到 [-1, 1] 的范围
            normalized_audio = audio_data / np.max(np.abs(audio_data))
            
            # 创建WAV文件
            with wave.open(output_path, 'w') as wav_file:
                wav_file.setnchannels(1)  # 设置声道数为1（单声道）
                wav_file.setsampwidth(2)  # 设置采样宽度为2字节（16位）
                wav_file.setframerate(sample_rate)  # 设置采样率
                wav_file.writeframes(normalized_audio.astype(np.float32).tobytes())
                
            print(f'Saved segment {i+1} of {file} as {output_file}')


if __name__=='__main__':
    # 设置数据文件夹路径、输出文件夹路径、段长度、WAV文件长度和采样率
    data_folder = '数据文件夹路径'
    output_folder = '输出文件夹路径'
    segment_length = 700000
    wav_length = 4  # WAV文件长度（单位：秒）
    sample_rate = 16000  # 采样率（例如：44100 Hz）

    # 调用函数进行处理并保存为WAV文件
    write_wav_files(data_folder, output_folder, segment_length, wav_length, sample_rate)