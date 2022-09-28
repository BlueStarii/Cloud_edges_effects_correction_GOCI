# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:31:53 2021
compare satellite with in-situ data
@author: Administrator
"""
import pandas as pd
import numpy as np
import h5py
import os 
import glob

# from L2wei_QA import QAscores_8Bands
from L2_flags import L3_mask

#load in-situ data
path = r'./Insitu/insitu_extract.csv'
insitu = pd.read_csv(path)
data = np.empty([1,19])
length = len(insitu)
valid_sat = []
#satellite path
path_sat = r'H:\DATA\GOCI\In-situ_data\SeaDAS'

for i in range(length):
    aqua_name = insitu['GOCI_file'][i]
    
    lat = insitu['lat'][i]
    lon = insitu['lon'][i]
    print('-------------No.',i,lat,lon,'---------------')
    try:
        satellite = h5py.File(glob.glob(os.path.join(path_sat,aqua_name+'*.L2_LAC_OC'))[0],'r')
        
        latitude = np.array(satellite['/navigation_data/latitude'])
        longitude = np.array(satellite['navigation_data/longitude'])

        [h,w] = latitude.shape
        #find closest pixels
        loc_err = abs(latitude-lat)+abs(longitude-lon)
        x,y = np.where(loc_err==np.min(loc_err))

        #3*3 box
        x1 = x-1
        x2 = x+2
        y1 = y-1
        y2 = y+2
    
        if x1 <0:
            x1 = [0]
        if x2 > h:
            x2 = h+1
        if y1 < 0:
            y1 = [0]
        if y2 > w:
            y2 = w+1
        
        # bands needed:412-667nm,senz, solz, sena, sola, l2_flags
        bands = ["Rrs_412","Rrs_443","Rrs_490","Rrs_555","Rrs_660","Rrs_680",\
                 "rhos_412","rhos_443","rhos_490","rhos_555","rhos_660","rhos_680",\
                "rhos_745","rhos_865","sena","senz","sola","solz"]
        img = np.empty([h,w,len(bands)])
        for j in range(len(bands)):
                dataset_band = satellite["/geophysical_data/" + bands[j]]
                value_band = dataset_band[:, :] * 1.
                # value_band[value_band == -32767.] = np.nan
                # value[:,i] = value_band.flatten()
                try:
                    gain = dataset_band.attrs["scale_factor"][0]
                    offset = dataset_band.attrs["add_offset"][0]
                except:
                    gain = 1
                    offset = 0
                img[:,:,j] = value_band*gain + offset
        
        l2flags = satellite['/geophysical_data/l2_flags'][x1[0]:x2[0],y1[0]:y2[0]]
        box = img[x1[0]:x2[0],y1[0]:y2[0],:]

        #l2_flags-valid pixels
        # flags = [0,1,3,4,5,6,8,9,10,12,14,15,16,19,20,21,22,24,25]
        flags = [0,1,3,4,8,9,14,16,25,26]
        # l2flags = np.array(value[:,-1], dtype='int32').transpose()
        # l2flags = box[:,:,-1].astype('int32')
        goodpixel = L3_mask(flags, l2flags)
        #将box转为二维数组，以便运算
        box_2d = np.empty([box.shape[0]*box.shape[1],box.shape[2]])
        for band in range(len(bands)):
            box_2d[:,band] = box[:,:,band].flatten()
            
        if np.sum(goodpixel)/((x2-x1)*(y2-y1)) >= 0.5:
            #cal for CV<0.15
            idx = np.argwhere(goodpixel.flatten()==1)
            valid_px = box_2d[idx.squeeze(),:]    #不在l2_flags内的valid pixel

            #计算前4个波段Rrs中位数是否小于0.15
            std = np.std(valid_px[:,:4],axis=0)
            mean = np.mean(valid_px[:,:4],axis=0)
            CV = np.median(std/mean)
            if CV<0.15:
                final_val = np.mean(valid_px,axis=0)
                final_val = np.append(final_val,i)    #添加每个像元的序列编号
                data = np.concatenate((data,final_val[np.newaxis,...]),axis=0)
                
                valid_sat.append(aqua_name)
            else:
                print('CV exceeds the threshold value',aqua_name)
        else:
            print('有效像元数量小于0.5：',aqua_name)
        
    except:
        print('数据打开失败或者转换失败')
data = np.delete(data,0,0)

#.观测角度用cosine处理，计算相对方位角

train_data = data[:,6:] #0-7：rrs, 8-21:rrc,22-26:geometry,27:class of water
train_label = data[:,:6]

train_dataset = train_data[:,:-1]
train_dataset[:,-3] = np.cos(train_data[:,-3])
train_dataset[:,-2] = np.cos(train_data[:,-1])
train_dataset[:,-1] = abs(train_data[:,-4]-train_data[:,-2])

idx1 = np.argwhere(train_dataset[:,8]>180)
column_ra = train_dataset[:,-1]
column_ra[idx1.squeeze()] = 360-column_ra[idx1.squeeze()]
train_dataset[:,-1] = column_ra
train_dataset[:,-1]=np.cos(train_dataset[:,-1])
print('data len:',len(train_dataset))

train_dataset = np.concatenate((train_dataset,data[:,-1][:,np.newaxis]),axis=1)

f = h5py.File('validation_insitu_SD.h5','w')
f.create_dataset('train',data=train_dataset)
f.create_dataset('SeaDAS',data=train_label) 
f.create_dataset('valid_sat',data=valid_sat)
f.close()       

