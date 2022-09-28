# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:20:08 2021
prediction for each image
1 l2flags mask
2 prediction

3 save for each scene

@author: Administrator
"""

# import os
import numpy as np
import h5py
import glob
import os

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
from L2_flags import L3_mask
from L2wei_QA import QAscores_6Bands

class myDataset(Dataset):

    def __init__(self,x,y)-> None:
        super().__init__()
        self.x_data = x
        self.y_data = y
    
    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        x_mean = np.array([0.05109411,0.04811902,0.0440184,0.03761746,0.02891396,0.02683822,\
                           0.02610199,0.02467448,-0.02693318,0.00343917,0.0355856], 'float32')
        x_std = np.array([0.01940052,0.01975982,0.01634109,0.02180259,0.01983091,0.0138185,\
                          0.01695971,0.01088832,0.6893147,0.7120903,0.7063683],'float32')
        y_mean = np.array([0.00637908,0.00584464,0.00540286,0.00324083,0.000782,0.00068278],'float32')
        y_std = np.array([0.00330699,0.00254451,0.00229242,0.00322304,0.00148145,0.00131122], 'float32')
        x = (x-x_mean)/x_std
        y = (y-y_mean)/y_std
        return x[..., np.newaxis].astype(np.float32).transpose([1, 0]), y[..., np.newaxis].astype(np.float32).transpose([1, 0])

    def __len__(self):
        return self.x_data.shape[0]
    
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
            # if i %50 == 0 and i != 0:
            #     predict = predict.cpu()
            #     print(str(i+1),'/',str(int(len(val_loader.dataset)/batch_size)))
            total_label = np.concatenate((total_label,y.cpu().detach().numpy().squeeze()), axis=0)
            total_predict = np.concatenate((total_predict,predict.cpu().detach().numpy().squeeze()),axis=0)
            i += 1
    total_label = np.delete(total_label,0,0)
    total_predict = np.delete(total_predict,0,0)
    
    y_mean = np.array([0.00637908,0.00584464,0.00540286,0.00324083,0.000782,0.00068278],'float32')
    y_std = np.array([0.00330699,0.00254451,0.00229242,0.00322304,0.00148145,0.00131122], 'float32')
        
    total_label = total_label*y_std+y_mean
    total_predict = total_predict*y_std+y_mean

    return total_label, total_predict

if __name__ == "__main__":
    n = 0
    path = r'D:\2-cloudremove\2-single_scene\G2019276'
    files = glob.glob(path+'\*.L2_LAC_OC')
    model = torch.load('best_model.pt')	
    batch_size = 10000
    for file in files:
        if os.path.exists(file[:-13]+'_Rrs.h5'):
            print(file)
        else:            
            '''image read'''
            n += 1
            print("---第",n,"景---")
            data = h5py.File(file,"r")
            longitude = data["/navigation_data/longitude"]
            [h,w] = longitude.shape
            latitude = data["/navigation_data/latitude"]
            geod = ['Rrs_412','Rrs_443','Rrs_490','Rrs_555','Rrs_660','Rrs_680',\
                    'rhos_412','rhos_443','rhos_490','rhos_555','rhos_660','rhos_680',\
                    'rhos_745','rhos_865',"sena","senz","sola","solz"] 
            value = np.empty((h*w, len(geod)),'float32')
            
            #2.维度转换（二维转一维），scale、add修正
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
                
            '''l2_flags mask'''
            flags = [0,1,3,4,5,6,10,12,14,15,16,19,20,21,22,24,25]
            flag_land = [1]
            # l2flags = np.array(value[:,-1], dtype='int32').transpose()
            l2flags = data["/geophysical_data/l2_flags"]
            value_masked = L3_mask(flags, l2flags)          #good pixel=1
            idx_land = L3_mask(flag_land, l2flags)          #extract land pixels
            delete_thick_cloud_idx = np.argwhere(value[:,-5]<0.06) #delete Rrc865>0.06
            val_index = np.argwhere(value_masked.flatten()==1)
            val_index = list(set(val_index.squeeze()).intersection(set(delete_thick_cloud_idx.squeeze())))
            data_masked = value[val_index,:]
            
            '''cal RAA'''
            train_data = data_masked[:,6:] #0-7：rrs, 8-21:rrc,22-26:geometry
            train_label = data_masked[:,:6]
    
            train_dataset = train_data[:,:-1]
            train_dataset[:,-3] = np.cos(train_data[:,-3])
            train_dataset[:,-2] = np.cos(train_data[:,-1])
            train_dataset[:,-1] = abs(train_data[:,-4]-train_data[:,-2])
    
            idx1 = np.argwhere(train_dataset[:,10]>180)
            column_ra = train_dataset[:,-1]
            column_ra[idx1.squeeze()] = 360-column_ra[idx1.squeeze()]
            train_dataset[:,-1] = column_ra
            train_dataset[:,-1]=np.cos(train_dataset[:,-1])
            
            '''prediction'''
            target_dt = myDataset(train_dataset,train_label)
            val_loader = DataLoader(target_dt, shuffle=False, batch_size=batch_size,
                                num_workers=0, drop_last=False, pin_memory=True)
            label, predict = evaluate(model, val_loader)
            
            '''QA'''                
            total_score_sd = []
            total_score_dl = []
            length_td = len(label)
            num_batch = round(length_td/10)
            test_lambda = np.array([412,443,490,555,660,680])

            for batch in range(10):
                if batch<9:
                    test_Rrs_sd = label[batch*num_batch:num_batch*(batch+1),:6]
                    test_Rrs_dl = predict[batch*num_batch:num_batch*(batch+1),:6]
                    maxCos_sd, cos_sd, clusterID_sd, totScore_sd = QAscores_6Bands(test_Rrs_sd, test_lambda)
                    maxCos_dl, cos_dl, clusterID_dl, totScore_dl = QAscores_6Bands(test_Rrs_dl, test_lambda)
                    total_score_sd = np.concatenate((total_score_sd,totScore_sd))
                    total_score_dl = np.concatenate((total_score_dl,totScore_dl))
                else:
                    test_Rrs_sd = label[batch*num_batch:,:6]
                    test_Rrs_dl = predict[batch*num_batch:,:6]
                    maxCos_sd, cos_sd, clusterID_sd, totScore_sd = QAscores_6Bands(test_Rrs_sd, test_lambda)
                    maxCos_dl, cos_dl, clusterID_dl, totScore_dl = QAscores_6Bands(test_Rrs_dl, test_lambda)
                    total_score_sd = np.concatenate((total_score_sd,totScore_sd))
                    total_score_dl = np.concatenate((total_score_dl,totScore_dl))
            
            '''back to 2-D image'''
            image_sd = np.zeros((h,w,6),'float32')
            image_dl = np.zeros((h,w,6),'float32')
            image_score_sd = np.zeros((h,w),'float32')
            image_score_dl = np.zeros((h,w),'float32')
            temp_sd = np.zeros((h*w,1),'float32')
            temp_dl = np.zeros((h*w,1),'float32')
            temp_score = np.zeros((h*w,1),'float32')
            '''save'''
            f = h5py.File(file[:-10]+'_Rrs.h5','w')
            for j in range(6):
                temp_sd[val_index,0] = label[:,j]
                temp_dl[val_index,0] = predict[:,j]
                
                image_sd[:,:,j] = temp_sd.reshape(h,w)
                image_dl[:,:,j] = temp_dl.reshape(h,w)
    
                f.create_dataset('seadas_'+str(j+1),data=image_sd[:,:,j])
                f.create_dataset('ACDL_'+str(j+1),data=image_dl[:,:,j])
            temp_score[val_index,0] = total_score_sd
            image_score_sd = temp_score.reshape(h,w)
            f.create_dataset('score_sd',data=image_score_sd)
            temp_score[val_index,0] = total_score_dl
            image_score_dl = temp_score.reshape(h,w)
            f.create_dataset('score_dl',data=image_score_dl)
            f.create_dataset('land',data=idx_land)
            f.close()
        
