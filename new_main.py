# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:00:52 2021

@author: Administrator
"""
import os
import numpy as np
import h5py
import time
# import math

# troch
import torch
import torch.nn as nn
# import torchvision
from torch.utils.data import Dataset
from tansformer import tnet
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from tools import EarlyStopping
from neuralnetwork import Net
# from TCN.mnist_pixel.model import TCN
# from torch.optim.lr_scheduler import StepLR

def huber(preds,target,delta=0.01):
    loss = torch.where(torch.abs(target-preds)>delta,0.5*((target-preds)**2),delta*torch.abs(target-preds)-0.5*(delta**2))
    return torch.mean(loss)

def loss_r2(preds,target):
    SST = (target-torch.mean(target,axis=0))**2
    SSR = (target-preds)**2
    R2_loss = SSR/SST
    return torch.mean(R2_loss)

def quantile_loss(preds, target, quantiles=[1,1,1,1,1,1,1,1]):
    assert not target.requires_grad
    assert preds.size(0) == target.size(0)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss

class myDataset(Dataset):

    def __init__(self, h5_path) -> None:
        super().__init__()

        file_ = h5py.File(h5_path)
        self.x_data = file_['train'][...]
        self.y_data = file_['label'][...]

    
    def __getitem__(self, index):

        x = self.x_data[index]
        y = self.y_data[index]
        '''mean-std标准化'''
        # x_mean = np.array([0.05109411,0.04811902,0.0440184,0.03761746,0.02891396,0.02683822,\
        #                    0.02610199,0.02467448,0.79587007,0.70632344,-0.38488677], 'float32')
        # x_std = np.array([0.01940052,0.01975982,0.01634109,0.02180259,0.01983091,0.0138185,\
        #                   0.01695971,0.01088832,0.07734057,0.12465386,0.44149712],'float32')
        # y_mean = np.array([0.00637908,0.00584464,0.00540286,0.00324083,0.000782,0.00068278],'float32')
        # y_std = np.array([0.00330699,0.00254451,0.00229242,0.00322304,0.00148145,0.00131122], 'float32')
        x_mean = np.array([4.94116545e-02, 4.72638085e-02, 4.43168543e-02, 3.88348363e-02,
                           2.79165041e-02, 2.57339198e-02, 2.43044309e-02, 2.30634790e-02,
                           3.67718849e+01, 4.10464516e+01, 1.20183235e+02], 'float32')
        x_std = np.array([1.8882811e-02, 1.9259019e-02, 1.6976848e-02, 2.3404231e-02,
                           1.9313632e-02, 1.3559640e-02, 1.6092913e-02, 1.0477692e-02,
                           8.8604574e+00, 1.3767522e+01, 4.0321960e+01], 'float32')
        y_mean = np.array([0.00653957, 0.00621547, 0.00604306, 0.00415713, 0.00102977,
                           0.00084897],'float32')
        y_std = np.array([0.00332638, 0.00254164, 0.0026375 , 0.00421917, 0.00175106,
                          0.00148965],'float32')
        x = (x-x_mean)/x_std
        y = (y-y_mean)/y_std
        '''log-mean-std标准化'''
        # # x[:8] = np.log10(x[:8])
        # y = np.log10(y)
        # # x_mean = np.array([-1.274941  , -1.2936296 , -1.3247117 , -1.3974781 , -1.5083703 ,
        # #                     -1.5253867 , -1.5411068 , -1.5462794 , -0.07524997, -0.04632621,
        # #                     0.03214116],'float32')
        # # x_std = np.array([0.14499204, 0.14552787, 0.13273062, 0.1731949 , 0.17417525,
        # #                     0.1570052 , 0.1780646 , 0.15630396, 0.7049453 , 0.7004588 ,
        # #                     0.69866544],'float32')
        # x_mean = np.array([0.05617052,0.05388187,0.04972308,0.04351444,0.03389908,0.032013,
        #                     0.03161354,0.03055517,-0.07524997,-0.04632621,0.03214116], 'float32')
        # x_std = np.array([0.01925436,0.01941399,0.01644007,0.01915223,0.01616032,0.0130647,
        #                   0.01589982,0.01179006,0.7049453,0.7004588,0.69866544],'float32')
        # y_mean = np.array([-2.2550328, -2.2472165, -2.263885 , -2.545079 , -3.2839222,-3.460463],'float32')
        # y_std = np.array([0.29616877, 0.1996771 , 0.15366554, 0.26256147, 0.40427145,0.59071046],'float32')
        # x = (x-x_mean)/x_std
        # y = (y-y_mean)/y_std
        
        '''
        归一化
        x_max = np.array([0.4,0.4,0.45,0.5,0.5,0.4,0.07,0.06,1,1,1],'float32')
        x_min = np.array([3.4768942e-05,  2.0540244e-04,  9.1665117e-03,  3.7500134e-03,\
                            1.3605136e-03,  6.7109773e-03,  7.2572781e-03,  2.0000016e-02,\
                           -9.9999982e-01, -9.9999958e-01, -1.0],'float32')
        y_max = np.array([0.041872, 0.034722, 0.058186, 0.037138, 0.02817, 0.029278],'float32')
        y_min = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001],'float32')
        x = (x-x_min)/(x_max-x_min)
        y = (y-y_min)/(y_max-y_min)
        '''
        return x[..., np.newaxis].astype(np.float32).transpose([1, 0]), y[..., np.newaxis].astype(np.float32).transpose([1, 0])
        # return x[..., np.newaxis].astype(np.float32), y[..., np.newaxis].astype(np.float32)
    def __len__(self):

        return self.x_data.shape[0]

# a = BroDataset(r'C:\Users\CaiNanDangDao\Documents\Tencent Files\648944084\FileRecv\train_data_demo.h5')
# print(a[0][0].shape, len(a))

def getLoss(index=0):
    '''
    这个我是参考的这里：https://www.jiqizhixin.com/articles/2018-06-21-3
    '''

    loss_list = [
        nn.MSELoss(),
        nn.SmoothL1Loss(),
        huber,
        nn.CrossEntropyLoss(),
        torch.cosh,
        quantile_loss # 这个没测试~
    ]
    if index > 6:
        return loss_list[0]
    else:
        return loss_list[index]

def train(train_loader,model,batch_size,opt,device='cuda'):
    
    
    model.train()
    # start_time = time.time()
    
    # scheduler = StepLR(opt, step_size=1, gamma=0.1)
    # 设置学习率：https://blog.csdn.net/qyhaill/article/details/103043637
    loss_func = getLoss(0)
    total_loss = 0.
    size = len(train_loader.dataset)
    batch = 0
    
    for x,y in train_loader:
        
        x = x.to(device)
        y = y.to(device)
        model.to(device)
        predict = model(x)
        loss = loss_func(predict, y)
        # loss1 = loss_func(predict[:,:,:6], y[:,:,:6])
        # loss2 = loss_func(predict[:,:,6:], y[:,:,6:])
        # loss = loss1 + loss2
        opt.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        total_loss += loss.item()
        log_interval = 50
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            # elapsed = time.time() - start_time
            predict = predict.cpu() 
            y = y.cpu()
            print('| epoch ',str(epoch),' | ',str(batch),'/',str(int(size/batch_size)),'batches | '
                  'lr',str(opt.param_groups[0]['lr']),' | loss ', str(cur_loss),
                  ' | predict ', predict.detach().numpy().squeeze()[0,0],' | label ', y.detach().numpy().squeeze()[0,0]) 
            total_loss = 0
            # start_time = time.time()
        batch += 1
    
    return model,cur_loss

def evaluate(model,val_loader,device='cuda'):
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    r2_total = 0.
    apd_all = 0.
    loss_func =  getLoss(0)
    with torch.no_grad():
        for x,y in val_loader:
            x = x.to(device)
            # y = y.to(device)
            predict = model(x)
            predict = predict.cpu()
            
            y_label = y.detach().numpy().squeeze()
            pre = predict.detach().numpy().squeeze()
            r2 = r2_score(y_label, pre)
            apd = np.mean((pre-y_label)/y_label)
            loss = loss_func(predict, y)
            total_loss += loss.item()
            r2_total += r2
            apd_all += apd
    return total_loss / (len(val_loader) - 1) ,r2_total/(len(val_loader) - 1),apd_all/(len(val_loader) - 1)
    

if __name__ == "__main__":
    path = r'train_ds_0726.h5'
    ds = myDataset(path)
    rate = 0.8 # 训练数据占总数据的比例
    batch_size = 3000
    epoch = 1000
    data_len = len(ds)
    
    patience = 10
    early_stopping = EarlyStopping(patience,  verbose=True)
    
    indices = torch.randperm(data_len).tolist()
    index = int(data_len * rate)
    train_ds = torch.utils.data.Subset(ds, indices[:index])
    val_ds = torch.utils.data.Subset(ds, indices[index:])
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size,
                                       num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, shuffle=True, batch_size=batch_size,
                                    num_workers=0, drop_last=True, pin_memory=True)
    model = tnet()
    # model = Net(11,6)
    # model = torch.load('best_model_716.pt')	
    # model = TCN(input_channels,n_classes,channel_sizes,kernel_size,dropout)
    opt = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.99)
    # opt = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, 
                                                            patience=8, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    for epoch in range(1,epoch+1):
        epoch_start_time = time.time()
        
        model,cur_loss = train(train_loader,model,batch_size,opt)
        val_loss,r2,apd = evaluate(model, val_loader)
        scheduler.step(cur_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.6f} | R2 {:5.2f}| APD {:5.2f} '
              .format(epoch, (time.time() - epoch_start_time),val_loss, r2,apd))
        print('-' * 89)
        early_stopping(val_loss, model)
        #满足 early stopping 要求
        if early_stopping.early_stop:
             print("Early stopping")
    		 # 结束模型训练
             break        

        
        
    

    


