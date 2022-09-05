# -*- coding: utf-8 -*-
import os
import base64
import xmltodict
import array
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#raw data
train_abnormal_dir = 'C:/Users/SPS/Desktop/심전도 공모전/electrocardiogram/data/train/arrhythmia/'
train_normal_dir = 'C:/Users/SPS/Desktop/심전도 공모전/electrocardiogram/data/train/normal/'
val_abnormal_dir = 'C:/Users/SPS/Desktop/심전도 공모전/electrocardiogram/data/validation/arrhythmia/'
val_normal_dir = 'C:/Users/SPS/Desktop/심전도 공모전/electrocardiogram/data/validation/normal/'
raw_data_dir_list = [train_abnormal_dir, train_normal_dir, val_abnormal_dir, val_normal_dir]

#save
train_abnormal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/train/arrhythmia/'
train_normal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/train/normal/'
val_abnormal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/validation/arrhythmia/'
val_normal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/validation/normal/'
save_data_dir_list = [train_abnormal_dir_save, train_normal_dir_save, val_abnormal_dir_save, val_normal_dir_save]
lead_id_list = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

for idx, dir_path in enumerate(raw_data_dir_list) :
    for file in os.listdir(dir_path):

        try:
            filename = os.fsdecode(file)
            if filename.endswith('.xml'):
                with open(dir_path + filename, 'rb') as xml:
                    ECG_temp = xmltodict.parse(xml.read().decode('utf8'))

                if (filename.startswith('6_')) or (filename.startswith('8_')):
                    lead_data = ECG_temp['RestingECG']['Waveform']['LeadData']
                elif (filename.startswith('5_')) :
                    lead_data = ECG_temp['RestingECG']['Waveform'][1]['LeadData']

                if (len(lead_data) == 8 or len(lead_data) == 12):
                    data_x_sequence = np.zeros([8, 5000])
                    data_x_sequence_new = np.zeros([5000, 8])
                    short_length = False

                    lead_data_count = -1
                    for j in lead_data:  # 다변량 시계열 데이터

                        lead_data_waveform = j['WaveFormData']
                        lead_id = j['LeadID']

                        if (lead_id in lead_id_list):
                            lead_b64 = base64.b64decode(lead_data_waveform)
                            rhythm = np.array(array.array('h', lead_b64))

                            if (len(rhythm) > 4990):
                                if len(rhythm) != 5000:
                                    if len(rhythm) > 5000:
                                        print('길이 5000초과 파일: ' + filename + ', 길이 :'+ str(len(rhythm)))
                                        rhythm = rhythm[:5000]
                                    elif len(rhythm) < 5000:
                                        print('길이 5000 미만 파일: ' + filename + ', 길이 :'+ str(len(rhythm)))
                                        difference = 5000 - len(rhythm)
                                        pad = np.zeros(difference)
                                        rhythm = np.append(rhythm, pad, 0)

                                lead_data_count += 1
                                data_x_sequence[lead_data_count] = rhythm
                            else :
                                print('길이부족파일 : '+ filename + ', 길이:'+  str(len(rhythm)))
                                short_length = True
                                break

                    # save to npy file
                    if (not short_length) :
                        data_x_sequence_new = data_x_sequence.T

                        np.save(save_data_dir_list[idx] + filename[:-4], data_x_sequence_new)

        except TypeError as e:
            print(e)
            print('에러 파일 : ' + filename)


















