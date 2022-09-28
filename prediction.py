# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:17:12 2021
predict Rrs using Rrc and geometries
@author: Administrator
"""
# import os
import numpy as np
import h5py
# import time
# import math

# troch
import torch
import torch.nn as nn
# import torchvision
from torch.utils.data import Dataset
from tansformer import tnet
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from tools import plot_scatter
from neuralnetwork import Net
from L2wei_QA import QAscores_6Bands
# from tools import EarlyStopping
from L2_flags import L3_mask
 
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
        x_mean = np.array([0.05109411,0.04811902,0.0440184,0.03761746,0.02891396,0.02683822,\
                           0.02610199,0.02467448,0.79587007,0.70632344,-0.38488677], 'float32')
        x_std = np.array([0.01940052,0.01975982,0.01634109,0.02180259,0.01983091,0.0138185,\
                          0.01695971,0.01088832,0.07734057,0.12465386,0.44149712],'float32')
        y_mean = np.array([0.00637908,0.00584464,0.00540286,0.00324083,0.000782,0.00068278],'float32')
        y_std = np.array([0.00330699,0.00254451,0.00229242,0.00322304,0.00148145,0.00131122], 'float32')

        x = (x-x_mean)/x_std
        y = (y-y_mean)/y_std
        
        '''log-mean-std标准化'''
        # x[:8] = np.log10(x[:8])
        # y = np.log10(y)
        # x_mean = np.array([-1.274941  , -1.2936296 , -1.3247117 , -1.3974781 , -1.5083703 ,
        #                     -1.5253867 , -1.5411068 , -1.5462794 , -0.07524997, -0.04632621,
        #                     0.03214116],'float32')
        # x_std = np.array([0.14499204, 0.14552787, 0.13273062, 0.1731949 , 0.17417525,
        #                     0.1570052 , 0.1780646 , 0.15630396, 0.7049453 , 0.7004588 ,
        #                     0.69866544],'float32')
        # x_mean = np.array([0.05617052,0.05388187,0.04972308,0.04351444,0.03389908,0.032013,
        #                     0.03161354,0.03055517,-0.07524997,-0.04632621,0.03214116], 'float32')
        # x_std = np.array([0.01925436,0.01941399,0.01644007,0.01915223,0.01616032,0.0130647,
        #                   0.01589982,0.01179006,0.7049453,0.7004588,0.69866544],'float32')
        # y_mean = np.array([-2.2550328, -2.2472165, -2.263885 , -2.545079 , -3.2839222,-3.460463],'float32')
        # y_std = np.array([0.29616877, 0.1996771 , 0.15366554, 0.26256147, 0.40427145,0.59071046],'float32')
        # x = (x-x_mean)/x_std
        # y = (y-y_mean)/y_std
        
        return x[..., np.newaxis].astype(np.float32).transpose([1, 0]), y[..., np.newaxis].astype(np.float32).transpose([1, 0])
    
    def __len__(self):

        return self.x_data.shape[0]
def getLoss(index=0):
    '''
    这个我是参考的这里：https://www.jiqizhixin.com/articles/2018-06-21-3
    '''

    loss_list = [
        nn.MSELoss(),
        nn.SmoothL1Loss(),
        torch.cosh,
        quantile_loss # 这个没测试~
    ]
    if index > 4:
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
            if i %20 == 0 and i != 0:
                predict = predict.cpu()
                print(str(i+1),'/',str(int(len(val_loader.dataset)/batch_size)))
            total_label = np.concatenate((total_label,y.cpu().detach().numpy().squeeze()), axis=0)
            total_predict = np.concatenate((total_predict,predict.cpu().detach().numpy().squeeze()),axis=0)
            i += 1
    total_label = np.delete(total_label,0,0)
    total_predict = np.delete(total_predict,0,0)

    # 标准化
    y_mean = np.array([0.00637908,0.00584464,0.00540286,0.00324083,0.000782,0.00068278],'float32')
    y_std = np.array([0.00330699,0.00254451,0.00229242,0.00322304,0.00148145,0.00131122], 'float32')
    
    # y_mean = np.array([-2.2550328, -2.2472165, -2.263885 , -2.545079 , -3.2839222,-3.460463],'float32')
    # y_std = np.array([0.29616877, 0.1996771 , 0.15366554, 0.26256147, 0.40427145,0.59071046],'float32')
    total_label = total_label*y_std+y_mean
    total_predict = total_predict*y_std+y_mean
    # total_label = 10**total_label
    # total_predict = 10**total_predict

    return total_label, total_predict

def metric(mat1,mat2):
    '''
    Parameters
    ----------
    mat1 : true.
    mat2 : predict.
    Returns: uncertainty, apd, r2
    '''
    ZX = (mat1-np.mean(mat1))/np.std(mat1)
    ZY = (mat2-np.mean(mat2))/np.std(mat2)
    r = np.sum(ZX*ZY)/(len(mat1))

    metrics = np.zeros([3])
    # uncertainty = np.std((mat2-mat1)/mat1)
    rpd = np.mean(abs(mat2-mat1)/mat1)
    # r2 = r2_score(mat1,mat2)
    bias = np.mean(mat2-mat1)                       
    
    # metrics[0] = uncertainty
    metrics[0] = rpd
    metrics[1] = r
    metrics[2] = bias
    return metrics

if __name__ == "__main__":
    path = r'test_ds.h5'
    dataset = h5py.File(path,'r')
    ds = myDataset(path)
    rate = 1 # 训练数据占总数据多少~
    batch_size = 8000
    data_len = len(ds)
    indices = torch.randperm(data_len).tolist()
    index = int(data_len * rate)
    val_ds = torch.utils.data.Subset(ds, indices[:index])
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size,
                                        num_workers=0, drop_last=False, pin_memory=True)

    model = torch.load('best_model.pt')	
    label, predict = evaluate(model, val_loader)
    f = h5py.File('test_predict.h5','w')
    f.create_dataset('predict',data=predict)
    f.create_dataset('label',data=label)
    f.create_dataset('cloud_Rrs',data=np.array(dataset['cloud_Rrs']))
    f.create_dataset('train',data=np.array(dataset['train']))
    f.close()
    # test_predict = h5py.File('test_predict.h5','r')
    # label = np.array(test_predict['label'])
    # predict = np.array(test_predict['predict'])
    
    sample_idx = np.random.randint(0,len(label),10000)
    x = label[sample_idx,:]
    y = predict[sample_idx,:]
    # '''delete thick cloud'''
    # test_Rrs = label
    # test_lambda = np.array([412,443,488,555,667,678])
    # maxCos, cos, clusterID, totScore = QAscores_6Bands(test_Rrs, test_lambda)
    # idx = np.argwhere(clusterID>5)
    # label = np.delete(label,idx.squeeze(),0)
    # predict = np.delete(predict,idx.squeeze(),0)
    
    band_name = np.array(['412nm','443nm','490nm','555nm','660nm','680nm'])
    stats = np.zeros([3,len(band_name)])
    for band in range(len(band_name)):
        stats[:,band] = metric(x[:,band], y[:,band])
        plot_scatter(x[:,band],y[:,band],stats[:,band],band_name[band])

        
    
    