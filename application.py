# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 09:25:18 2022

@author: Administrator
"""
import numpy as np
import h5py               #h5py 和 gdal不兼容
import glob
from osgeo import gdal,osr
import os
import torch
import torch.nn as nn
# import torchvision
from torch.utils.data import Dataset
from tansformer import tnet
from torch.utils.data import DataLoader
from L2_flags import L3_mask
from L2wei_QA import QAscores_6Bands

def T_chl(band1,band2,band3,band4):
    '''
    GOCI
    band1:412
    band2:443
    band3:490
    bang4:555
    '''
    R = (band2/band4)*(band1/band3)**-1.012
    chl = 10**(0.342-2.511*np.log10(R)-0.277*np.log10(R)**2)
    # ci = band2-0.5*(band1+band3)
    # # ci[np.argwhere(ci>0.000355)] = 0.000355   #ci指数小于0.000355的可用于计算chl
    # # chl = 10**(-0.4909+191.6590*ci)       #OCI1
    # chl = 10**(-0.4287+230.47*ci)         #OCI2
    chl[band1==0] = 0
    return chl

class myDataset(Dataset):

    def __init__(self, x,y) -> None:
        super().__init__()

        # file_ = h5py.File(h5_path)
        self.x_data = x
        self.y_data = y
    
    def __getitem__(self, index):

        x = self.x_data[index]
        y = self.y_data[index]
        '''mean-std标准化'''
        x_mean = np.array([0.05109411,0.04811902,0.0440184,0.03761746,0.02891396,0.02683822,\
                           0.02610199,0.02467448,-0.02693318,0.00343917,0.0355856], 'float32')
        x_std = np.array([0.01940052,0.01975982,0.01634109,0.02180259,0.01983091,0.0138185,\
                          0.01695971,0.01088832,0.6893147,0.7120903,0.7063683],'float32')
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
                # print(str(i+1),'/',str(int(len(val_loader.dataset)/batch_size)))
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

def QA_batch(label,predict,w,h,val_index):
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
    # image_score_sd = np.zeros((h,w),'float32')
    # image_score_dl = np.zeros((h,w),'float32')
    temp_sd = np.zeros((h*w,1),'float32')
    temp_dl = np.zeros((h*w,1),'float32')
    temp_score_sd = np.zeros((h*w,1),'float32')
    temp_score_dl = np.zeros((h*w,1),'float32')
    '''save'''
    temp_score_sd[val_index,0] = total_score_sd
    # image_score_sd = temp_score.reshape(h,w)
    temp_score_dl[val_index,0] = total_score_dl
    # image_score_dl = temp_score.reshape(h,w)
    for j in range(6):
        temp_sd[val_index,0] = label[:,j]
        temp_dl[val_index,0] = predict[:,j]
        temp_sd[temp_score_sd<1] = 0
        temp_dl[temp_score_dl<1] = 0
        image_sd[:,:,j] = temp_sd.reshape(h,w)
        image_dl[:,:,j] = temp_dl.reshape(h,w)
    return image_sd,image_dl

def dl_predict(data,model):
    batch_size = 10000
    '''image read'''
    # print("---第",n,"景---")
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
    # flag_land = [1]
    # l2flags = np.array(value[:,-1], dtype='int32').transpose()
    l2flags = data["/geophysical_data/l2_flags"]
    value_masked = L3_mask(flags, l2flags)          #good pixel=1
    # idx_land = L3_mask(flag_land, l2flags)          #extract land pixels
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
    
    image_sd, image_dl = QA_batch(label, predict,w,h,val_index)
            
    return image_sd, image_dl

def T_tsm(band1,band2,band3,band4):
    '''
    band1:490
    band2:555
    band3:660
    band4:680
    '''
    TSM = 10**(0.649+25.623*(band2+(band3+band4)/2)-0.646*(band1/band2))
    
    return TSM

def write_bands(im_data, banddes=None):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_height, im_width, im_bands = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    # 数据类型必须有，因为要计算需要多大内存空间
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create("", im_width, im_height, im_bands, datatype)

    # 写入数组数据
    if im_bands == 1:
        # dataset.GetRasterBand(1).SetNoDataValue(65535)
        try:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入
        except:
            dataset.GetRasterBand(1).WriteArray(im_data[:,:,0])
    else:
        # if banddes==None:
        # banddes = ['Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_520', 'Rrs_565', 'Rrs_670', 'chlor_a']
        for i in range(im_bands):
            try:
                # dataset.GetRasterBand(i + 1).SetNoDataValue(65535)
                RasterBand = dataset.GetRasterBand(i + 1)
                # RasterBand.SetDescription(banddes[i])
                RasterBand.WriteArray(im_data[:, :, i])
            except IndentationError:
                print('band:'+i)

    return dataset

def modis_gcp(longitude,latitude,inpt,outpt,label,radi_bands=['chlor_a'], latlon=False):
    if latlon:
        value = np.empty((longitude.shape[0], longitude.shape[1], len(radi_bands) + 2))
    else:
        value = np.empty((longitude.shape[0], longitude.shape[1], len(radi_bands)))
    try:
        # 如果需要加入经纬度数据
        # value[:, :, len(radi_bands)-1] = inpt
        # value[:, :, len(radi_bands)] = latitude
        # value[:, :, len(radi_bands) + 1] = longitude
        value = inpt
    except:
        pass        
    # 将波段数据写入临时内存文件
    image: gdal.Dataset = write_bands(value)
    # 控制点列表, 设置7*7个控制点
    gcps = []
    x_arr = np.linspace(0, longitude.shape[1] - 1, num=7, endpoint=True, dtype=np.int)
    y_arr = np.linspace(0, longitude.shape[0] - 1, num=7, endpoint=True, dtype=np.int)
    for x in x_arr:
        for y in y_arr:
            if abs(longitude[y, x]) > 180 or abs(latitude[y, x]) > 90:
                continue
            gcps.append(gdal.GCP(np.float64(longitude[y, x]), np.float64(latitude[y, x]),
                                 0,
                                 np.float64(x), np.float64(y)))
    # 设置空间参考
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    sr.MorphToESRI()
    # 给数据及设置控制点及空间参考
    image.SetGCPs(gcps, sr.ExportToWkt())

    outfile = os.path.dirname(outpt) + os.sep + os.path.basename(outpt) + label+'.tif'

    # 校正
    if latlon:
        cutlinelayer = radi_bands+['latitude', 'longitude']
    else:
        cutlinelayer = radi_bands
    dst = gdal.Warp(outfile, image, format='GTiff', tps=True, xRes=0.01, yRes=0.01, dstNodata=np.nan,
              resampleAlg=gdal.GRA_NearestNeighbour, cutlineLayer=cutlinelayer)  # dstNodata=65535

    for i,bandname in enumerate(cutlinelayer):
        band = dst.GetRasterBand(i+1)
        band.SetMetadata({'bandname':bandname})
        band.SetDescription(bandname)
    image: None
    return outfile


if __name__=='__main__':
    path = r'D:\2-cloudremove\5-daily\SeaDAS'
    files = glob.glob(path+'\*.L2_LAC_OC')
    model = torch.load('best_model.pt')	
    # batch_size = 10000
    for file in files:
        print(file)
        data = h5py.File(file,"r")
        lon = np.array(data["/navigation_data/longitude"])
        lat = np.array(data["/navigation_data/latitude"])
        image_sd, image_dl = dl_predict(data,model)
        # chl_sd = CI(image_sd[:,:,1],image_sd[:,:,4],image_sd[:,:,5])
        chl_dl = T_chl(image_dl[:,:,0],image_dl[:,:,1],image_dl[:,:,2],image_dl[:,:,3])
        TSM_dl = T_tsm(image_dl[:,:,2],image_dl[:,:,3],image_dl[:,:,4],image_dl[:,:,5])
        # modis_gcp(lon,lat,chl_sd,file[:-7],'sd_chl',radi_bands=['chlor_a'])
        modis_gcp(lon,lat,chl_dl,file[:-10],'dl_chl',radi_bands=['chlor_a'])
        modis_gcp(lon,lat,TSM_dl,file[:-10],'dl_tsm',radi_bands=['chlor_a'])
