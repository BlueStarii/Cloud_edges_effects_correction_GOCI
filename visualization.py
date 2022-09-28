# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 17:47:17 2021

visualization of image after cloud remove

@author: Menjl
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
import sys
sys.path.append("../../")
from neuralnetwork import Net
from L2_flags import L3_mask
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

    def __init__(self, train,label) -> None:
        super().__init__()

        # file_ = h5py.File(h5_path)
        self.x_data = train
        self.y_data = label

    
    def __getitem__(self, index):

        x = self.x_data[index]
        y = self.y_data[index]
        '''log-mean-std标准化'''
        # x[:8] = np.log10(x[:8])
        y = np.log10(y)
        # x_mean = np.array([-1.274941  , -1.2936296 , -1.3247117 , -1.3974781 , -1.5083703 ,
        #                     -1.5253867 , -1.5411068 , -1.5462794 , -0.07524997, -0.04632621,
        #                     0.03214116],'float32')
        # x_std = np.array([0.14499204, 0.14552787, 0.13273062, 0.1731949 , 0.17417525,
        #                     0.1570052 , 0.1780646 , 0.15630396, 0.7049453 , 0.7004588 ,
        #                     0.69866544],'float32')
        x_mean = np.array([0.05617052,0.05388187,0.04972308,0.04351444,0.03389908,0.032013,
                            0.03161354,0.03055517,-0.07524997,-0.04632621,0.03214116], 'float32')
        x_std = np.array([0.01925436,0.01941399,0.01644007,0.01915223,0.01616032,0.0130647,
                          0.01589982,0.01179006,0.7049453,0.7004588,0.69866544],'float32')
        y_mean = np.array([-2.2550328, -2.2472165, -2.263885 , -2.545079 , -3.2839222,-3.460463],'float32')
        y_std = np.array([0.29616877, 0.1996771 , 0.15366554, 0.26256147, 0.40427145,0.59071046],'float32')
        x = (x-x_mean)/x_std
        y = (y-y_mean)/y_std
        
        return x[..., np.newaxis].astype(np.float32).transpose([1, 0]), y[..., np.newaxis].astype(np.float32).transpose([1, 0])
        # return x[..., np.newaxis].astype(np.float32), y[..., np.newaxis].astype(np.float32)
    def __len__(self):

        return self.x_data.shape[0]

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
def evaluate(model,val_loader,device='cuda'):
    model.eval() # Turn on the evaluation mode   
    total_label = np.zeros([1,6])
    total_predict = np.zeros([1,6])
    i = 0
    with torch.no_grad():
        for x,y in val_loader:
            
            x = x.to(device)
            y = y.to(device)
            model.to(device)
            predict = model(x)
            if i %10 == 0 and i != 0:
                predict = predict.cpu()
                print(str(i+1),'/',str(int(len(val_loader.dataset)/batch_size)))
            total_label = np.concatenate((total_label,y.cpu().detach().numpy().squeeze()), axis=0)
            total_predict = np.concatenate((total_predict,predict.cpu().detach().numpy().squeeze()),axis=0)
            i += 1
    total_label = np.delete(total_label,0,0)
    total_predict = np.delete(total_predict,0,0)
    y_mean = np.array([-2.2550328, -2.2472165, -2.263885 , -2.545079 , -3.2839222,-3.460463],'float32')
    y_std = np.array([0.29616877, 0.1996771 , 0.15366554, 0.26256147, 0.40427145,0.59071046],'float32')
    total_label = total_label*y_std+y_mean
    total_predict = total_predict*y_std+y_mean
    total_label = 10**total_label
    total_predict = 10**total_predict

    return total_label, total_predict

if __name__ == "__main__":
    path = r'D:\2-cloudremove\5-daily\SeaDAS\G2019276001535.L2_LAC_OC'
    data = h5py.File(path,'r')
    l2_flags = data['/geophysical_data/l2_flags']
    geod = ['Rrs_412','Rrs_443','Rrs_490','Rrs_555','Rrs_660','Rrs_680',\
            'rhos_412','rhos_443','rhos_490','rhos_555','rhos_660','rhos_680',\
            'rhos_745','rhos_865',"sena","senz","sola","solz"]                     
    value = np.empty((l2_flags.shape[0]*l2_flags.shape[1], len(geod)),'float32')
    for i in range(len(geod)):
            dataset_band = data["/geophysical_data/" + geod[i]]
            value_band = dataset_band[:, :] * 1.
            # value_band[value_band == -32767.] = np.nan
            value[:,i] = value_band.flatten()
            try:
                gain = dataset_band.attrs["scale_factor"][0]
                offset = dataset_band.attrs["add_offset"][0]
            except:
                gain = 1
                offset = 0
            value[:,i] = value[:,i]*gain + offset
    '''calculate RAA '''
    train_data = value[:,6:] #0-5：rrs, 6-13:rrc,14-17:geometries
    train_label = value[:,:6]
    
    train_dataset = train_data[:,:-1]
    train_dataset[:,-3] = np.cos(train_data[:,-3])
    train_dataset[:,-2] = np.cos(train_data[:,-1])
    train_dataset[:,-1] = abs(train_data[:,-4]-train_data[:,-2])
    
    idx1 = np.argwhere(train_dataset[:,10]>180)
    column_ra = train_dataset[:,-1]
    column_ra[idx1.squeeze()] = 360-column_ra[idx1.squeeze()]
    train_dataset[:,-1] = column_ra
    train_dataset[:,-1]=np.cos(train_dataset[:,-1])
    
    '''mask'''
    l2_flags = data['/geophysical_data/l2_flags']
    flags = [0,1,3,4,5,6,10,12,14,15,16,19,20,21,22,24,25]
    goodpixels = L3_mask(flags, l2_flags)
    thick_cloud = np.where(np.array(data['/geophysical_data/rhos_865'])>0.06)# 删除厚云
    goodpixels[thick_cloud] = 0
    val_index = np.argwhere(goodpixels.flatten()==1).squeeze()    #good pixels index
    
    '''prediction '''
    ds = myDataset(train_dataset[val_index,:],train_label[val_index,:])
    rate = 1 # 训练数据占总数据多少~
    batch_size = 5000
    data_len = len(ds)
    indices = torch.randperm(data_len).tolist()
    index = int(data_len * rate)
    val_ds = torch.utils.data.Subset(ds, indices[:index])
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size,
                                       num_workers=0, drop_last=False, pin_memory=True)

    # model = tnet()
    # model = Net(11,8)
    model = torch.load('best_model.pt')	
    label, predict = evaluate(model, val_loader)
    
    predictions = np.zeros((l2_flags.shape[0]*l2_flags.shape[1],6),'float32')
    labels = np.zeros((l2_flags.shape[0]*l2_flags.shape[1],6),'float32')
    
    predict_images = np.zeros((l2_flags.shape[0],l2_flags.shape[1],6),'float32')
    label_images = np.zeros((l2_flags.shape[0],l2_flags.shape[1],6),'float32')
    
    predictions[val_index,:] = predict
    labels[val_index,:] = label
    
    for i in range(6):
        predict_images[:,:,i] = predictions[:,i].reshape([l2_flags.shape[0],l2_flags.shape[1]])
        label_images[:,:,i] = labels[:,i].reshape([l2_flags.shape[0],l2_flags.shape[1]])

    file_name = 'predict_vis1.h5'
    f = h5py.File(file_name,'w')
    f.create_dataset('predict_412',data=predict_images[:,:,0])
    f.create_dataset('predict_443',data=predict_images[:,:,1])
    f.create_dataset('predict_490',data=predict_images[:,:,2])
    f.create_dataset('predict_555',data=predict_images[:,:,3])
    f.create_dataset('predict_660',data=predict_images[:,:,4])
    f.create_dataset('predict_680',data=predict_images[:,:,5])
    f.create_dataset('SeaDAS_412',data=label_images[:,:,0])
    f.create_dataset('SeaDAS_443',data=label_images[:,:,1])
    f.create_dataset('SeaDAS_490',data=label_images[:,:,2])
    f.create_dataset('SeaDAS_555',data=label_images[:,:,3])
    f.create_dataset('SeaDAS_660',data=label_images[:,:,4])
    f.create_dataset('SeaDAS_680',data=label_images[:,:,5])
    f.create_dataset('l2_flags',data=l2_flags)
    f.close()







