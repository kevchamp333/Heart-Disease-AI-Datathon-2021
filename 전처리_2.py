import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

train_abnormal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/train/arrhythmia/'
train_normal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/train/normal/'
val_abnormal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/validation/arrhythmia/'
val_normal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/validation/normal/'
columns = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

#train_abnormal
train_abnormal_file = os.listdir(train_abnormal_dir_save)
                                                            # 파일 개수                     행                 열
train_abnormal_data_x = np.zeros((len(train_abnormal_file), 5000, len(columns)))
for idx, filename in enumerate(train_abnormal_file):
    data = np.load(train_abnormal_dir_save + filename[:-4] + '.npy')
    train_abnormal_data_x[idx] = data
    print('train abnormal : '+ str(idx))

np.save('C:/Users/SPS/Desktop/심전도 공모전/preprocessing_2/train_abnormal.npy', train_abnormal_data_x)

# train_normal
train_normal_file = os.listdir(train_normal_dir_save)
                                                            # 파일 개수                     행                 열
train_normal_data_x = np.zeros((len(train_normal_file), 5000, len(columns)))
for idx, filename in enumerate(train_normal_file):
    data = np.load(train_normal_dir_save + filename[:-4] + '.npy')
    train_normal_data_x[idx] = data
    print('train normal : ' + str(idx))

np.save('C:/Users/SPS/Desktop/심전도 공모전/preprocessing_2/train_normal.npy', train_normal_data_x)

#val_abnormal
val_abnormal_file = os.listdir(val_abnormal_dir_save)
                                                            # 파일 개수                     행                 열
val_abnormal_data_x = np.zeros((len(val_abnormal_file), 5000, len(columns)))
for idx, filename in enumerate(val_abnormal_file):
    data = np.load(val_abnormal_dir_save + filename[:-4] + '.npy')
    val_abnormal_data_x[idx] = data
    print('val abnormal : ' + str(idx))
np.save('C:/Users/SPS/Desktop/심전도 공모전/preprocessing_2/val_abnormal.npy', val_abnormal_data_x)

# val_normal
val_normal_file = os.listdir(val_normal_dir_save)
                                                            # 파일 개수                     행                 열
val_normal_data_x = np.zeros((len(val_normal_file), 5000, len(columns)))
for idx, filename in enumerate(val_normal_file):
    data = np.load(val_normal_dir_save + filename[:-4] + '.npy')
    val_normal_data_x[idx] = data
    print('val normal : ' + str(idx))

np.save('C:/Users/SPS/Desktop/심전도 공모전/preprocessing_2/val_normal.npy', val_normal_data_x)
