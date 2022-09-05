# -*- coding: utf-8 -*-
"""
Created on 2021/12/12
@author: Yoo Jaejin, Hwang Wooyoung, Lee Yujin
"""

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv7x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, dilation=1):
        super(BasicBlock, self).__init__()

        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        self.planes = planes
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv7x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(planes, planes // 16)
        self.fc2 = nn.Linear(planes // 16, planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        #Squeeze and Excitation block
        SE = self.pooling(out)
        SE = SE.view(len(SE), self.planes)
        SE = self.fc1(SE)
        SE = self.fc2(SE)
        SE = self.sigmoid(SE)

        out = out * SE.view(len(SE), self.planes, 1)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False, groups=1):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.conv1 = nn.Conv1d(8, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.adapavgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride), nn.BatchNorm1d(planes))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups))

        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adapavgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

# =============================================================================
# # train model
# =============================================================================
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

#function to calculate metric per mini-batch
def metric_batch(output, target, val = 'train'):
    pred = (output >= 0.5).type(torch.int)
    corrects = pred.eq(target.view_as(pred)).sum().item()

    if val != 'train':
        print('output: {}, target: {}, \n pred: {}, corrects: {}'.format(output, target, pred, corrects))

    return corrects

def loss_batch(loss_func, output, target, opt=None, val='train'):
    if opt is not None:
        target = torch.unsqueeze(target, 1)
    else:
        target = target.view((1, 1))

    loss = loss_func(output, target) # 배치 단위의 loss
    metric_b = metric_batch(output, target, val)

    print("correct : " + str(metric_b) + "/" + str(output.shape[0]))
    print("loss : " + str(loss.item()))

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

#function to calculate loss and metric per epoch
def loss_epoch(epoch, model, loss_func, data_dl, label, sanity_check = False, opt = None, val = 'train'):
    running_loss = 0.0
    running_metric_b = 0.0
    len_data = len(data_dl)

    if val == 'train':
        for i, data in enumerate(data_dl):    
            xb = torch.tensor(data).view(17, 8, 5000).float().to(device) # 1배치에 17개씩
            yb = torch.tensor(label[i]).float().to(device)

            output = model(xb)
            loss_b, metric_b = loss_batch(loss_func, output, yb, opt, val)
            running_loss += loss_b

            if metric_b is not None:
                running_metric_b += metric_b
    elif val == 'val' : # validation
        val_pred = np.zeros(label.shape)
        for i, data in enumerate(data_dl):
            xb = torch.tensor(data_dl[i]).view(1, 8, 5000).float().to(device)
            yb = torch.tensor(label[i]).float().to(device)
            output = model(xb)
            val_pred[i] = output

            loss_b, metric_b = loss_batch(loss_func, output, yb, opt, val)
            running_loss += loss_b

            if metric_b is not None:
                running_metric_b += metric_b

        # Validation AUC-ROC curve
        fpr, tpr, _ = roc_curve(label, val_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title('ROC AUC Curve', fontsize=20)
        plt.legend(loc="lower right", fontsize=20)
        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", f"ROC_Curve_Val_result_{epoch}.png"))
        plt.clf()
        plt.close()

    elif val == 'test' : # test
        val_pred = np.zeros(label.shape)
        for i, data in enumerate(data_dl):
            xb = torch.tensor(data_dl[i]).view(1, 8, 5000).float().to(device)
            yb = torch.tensor(label[i]).float().to(device)
            output = model(xb)
            val_pred[i] = output

            loss_b, metric_b = loss_batch(loss_func, output, yb, opt, val)
            running_loss += loss_b

            if metric_b is not None:
                running_metric_b += metric_b

        # Validation AUC-ROC curve
        fpr, tpr, _ = roc_curve(label, val_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title('ROC AUC Curve', fontsize=20)
        plt.legend(loc="lower right", fontsize=20)
        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", f"ROC_Curve_Test_result_{epoch}.png"))
        plt.clf()
        plt.close()

    if val == 'train':
        loss = running_loss / (len_data  * data_dl.shape[1])
        metric = running_metric_b / (len_data  * data_dl.shape[1])
    else :
        loss = running_loss / len_data
        metric = running_metric_b / len_data  # 전체 데이터 개수 중에 맞은 데이터 개수

    return loss, metric

def train_val(model, params):

    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    opt=params['optimizer']
    train_dl=params['train_dl']
    train_label = params['train_label']
    val_dl=params['val_dl']
    val_label = params['val_label']
    sanity_check=params['sanity_check']
    lr_scheduler=params['lr_scheduler']
    path2weights = params['path2weights']

    loss_history = {'train loss': [], 'val loss': []}
    metric_history = {'train accuracy': [], 'val accuracy': []}

    best_loss = float('inf')
    start_time = time.time()
    epoch_list = []
    for epoch in range(num_epochs):
        epoch_list.append(epoch +1)
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))

        # train
        model.train()
        val = False

        train_loss, train_metric = loss_epoch(epoch, model, loss_func, train_dl, train_label, sanity_check, opt, val = 'train')
        loss_history['train loss'].append(train_loss) # loss
        metric_history['train accuracy'].append(train_metric) # 정확도

        # validation
        model.eval()
        val = True
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(epoch, model, loss_func, val_dl, val_label, sanity_check, val = 'val')

        loss_history['val loss'].append(val_loss)
        metric_history['val accuracy'].append(val_metric)
        lr_scheduler.step(val_loss)

        print('train loss: %.6f, val loss: %.6f, val_accuracy: %.2f, time: %.4f min' % (train_loss, val_loss, val_metric, (time.time() - start_time) / 60))
        print('-' * 10)

        #accuracy plot
        plt.figure(figsize=(10, 10))
        plt.plot(epoch_list, metric_history['train accuracy'], label='train accuracy', linewidth=1.5)
        plt.plot(epoch_list, metric_history['val accuracy'], label='val accuracy', linewidth=1.5)
        plt.xlabel('Epoch', fontsize=20)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.legend(loc='best', fontsize=18)
        plt.subplots_adjust(bottom=0.3)

        accu_name = ['train accuracy', 'val accuracy']
        for value in accu_name :
            accu_list = np.round(metric_history[value], 2) # accu
            for v in range(len(accu_list)):
                if (v % 1 == 0):
                    plt.text(epoch_list[v], accu_list[value][v], accu_list[value][v], fontsize=12, horizontalalignment='center', verticalalignment='bottom')

        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", f"Accu_result_{epoch}.png"))
        plt.clf()
        plt.close()

        #loss plot
        plt.figure(figsize=(10, 10))
        plt.plot(epoch_list, loss_history['train loss'], label='train loss', linewidth=1.5)
        plt.plot(epoch_list, loss_history['val loss'], label='val loss', linewidth=1.5)
        plt.xlabel('Epoch', fontsize=20)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.legend(loc='best', fontsize=18)
        plt.subplots_adjust(bottom=0.3)

        accu_name = ['train loss', 'val loss']
        for value in accu_name:
            loss_list = np.round(loss_history[value], 2)  # accu
            for v in range(len(loss_list)):
                if (v % 1 == 0):
                    plt.text(epoch_list[v], loss_list[value][v], loss_list[value][v], fontsize=12,
                             horizontalalignment='center', verticalalignment='bottom')

        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", f"Loss_result_{epoch}.png"))
        plt.clf()
        plt.close()

        if val_loss < best_loss:  # epoch마다 val loss 측정해서 이전보다 작을때 weight 저장
            best_loss = val_loss
            torch.save(model.state_dict(), path2weights + 'ECG_model_' + str(epoch) +'.pt')
            print('Copied best model weights!')
            print('Get best val_loss')

    return model, loss_history, metric_history

def test(model, params):
    loss_func = params['loss_func']
    test_dl = params['test_dl']
    test_label = params['test_label']
    sanity_check = params['sanity_check']
    path2weights = params['path2weights']
    start_time = time.time()

    # test
    model.load_state_dict(torch.load(path2weights))
    model.eval()
    with torch.no_grad():
        test_loss, test_metric = loss_epoch(1, model, loss_func, test_dl, test_label, sanity_check, val='test')

    print('val loss: %.6f, val_accuracy: %.2f, time: %.4f min' % (test_loss, test_metric, (time.time() - start_time) / 60))
    print('-' * 10)

    return model

# =============================================================================
# # Get data
# =============================================================================
#train data 디렉토리
train_abnormal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/train_abnormal.npy'
train_normal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/train_normal.npy'
#validation data 디렉토리
val_abnormal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/val_abnormal.npy'
val_normal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/val_normal.npy'
# test data 디렉토리
test_abnormal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/test_abnormal.npy'
test_normal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/test_normal.npy'

test_mode = True

if (test_mode) :
    # set test data
    test_x_abnormal = np.load(test_abnormal_dir_save)
    test_x_normal = np.load(test_normal_dir_save)
    test_x_abnormal_len = test_x_abnormal.shape[0]
    test_x_normal_len = test_x_normal.shape[0]
    data_x_test_total = np.append(test_x_abnormal, test_x_normal, axis=0)
    del test_x_abnormal
    del test_x_normal

    shuff_indx = np.random.permutation(len(data_x_test_total))
    data_x_test_total = data_x_test_total[shuff_indx, :]

    data_y_test_abnormal = np.full(test_x_abnormal_len, 1)  # 비정상 : 1
    data_y_test_normal = np.full(test_x_normal_len, 0)  # 정상 0
    data_y_test_total = np.concatenate((data_y_test_abnormal, data_y_test_normal), axis=0)
    del data_y_test_abnormal
    del data_y_test_normal
    data_y_test_total = data_y_test_total[shuff_indx]

else :
    # set train data
    train_x_abnormal = np.load(train_abnormal_dir_save)
    train_x_normal = np.load(train_normal_dir_save)
    train_x_abnormal_len = train_x_abnormal.shape[0]
    train_x_normal_len = train_x_normal.shape[0]
    data_x_train_total = np.append(train_x_abnormal, train_x_normal, axis=0)
    del train_x_abnormal
    del train_x_normal

    shuff_indx = np.random.permutation(len(data_x_train_total))
    data_x_train_total = data_x_train_total[shuff_indx, :]

    data_y_abnormal = np.full(train_x_abnormal_len, 1)  # 비정상 : 1
    data_y_normal = np.full(train_x_normal_len, 0)  # 정상 0
    data_y_train_total = np.concatenate((data_y_abnormal, data_y_normal), axis=0)
    del data_y_abnormal
    del data_y_normal
    data_y_train_total = data_y_train_total[shuff_indx]
    data_x_train_total.shape = (2287, 17, 5000, 8)
    data_y_train_total.shape = (2287, 17)

    # set validation data
    val_x_abnormal = np.load(val_abnormal_dir_save)
    val_x_normal = np.load(val_normal_dir_save)
    val_x_abnormal_len = val_x_abnormal.shape[0]
    val_x_normal_len = val_x_normal.shape[0]
    data_x_val_total = np.append(val_x_abnormal, val_x_normal, axis=0)
    del val_x_abnormal
    del val_x_normal

    shuff_indx = np.random.permutation(len(data_x_val_total))
    data_x_val_total = data_x_val_total[shuff_indx, :]

    data_y_val_abnormal = np.full(val_x_abnormal_len, 1)  # 비정상 : 1
    data_y_val_normal = np.full(val_x_normal_len, 0)  # 정상 0
    data_y_val_total = np.concatenate((data_y_val_abnormal, data_y_val_normal), axis=0)
    del data_y_val_abnormal
    del data_y_val_normal
    data_y_val_total = data_y_val_total[shuff_indx]

# =============================================================================
# #initialize model
# =============================================================================
model = ResNet(BasicBlock, [3, 4, 6, 3]).to(device)
loss_func = nn.BCELoss()
opt = optim.Adam(model.parameters(), lr = 0.001)
lr_scheduler = ReduceLROnPlateau(opt, mode = 'min', factor = 0.1, patience = 10)
path2weights = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/코드/model/ECG_model.pt' #모델 가중치 저장 위치
path3weights = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/코드/model/ECG_model.pt' #모델 가중치 불러오는 위치
epoch = 100

if (test_mode) :
    params_train = {
        'loss_func': loss_func,
        'test_dl': data_x_test_total,
        'test_label': data_y_test_total,
        'sanity_check': False,
        'path2weights': path3weights
    }
    model = test(model, params_train)
else :
    params_train = {
        'num_epochs': epoch,
        'optimizer': opt,
        'loss_func': loss_func,
        'train_dl': data_x_train_total,
        'train_label': data_y_train_total,
        'val_dl': data_x_val_total,
        'val_label': data_y_val_total,
        'sanity_check': False,
        'lr_scheduler': lr_scheduler,
        'path2weights': path2weights
    }
    model, loss_hist, metric_hist = train_val(model, params_train)


