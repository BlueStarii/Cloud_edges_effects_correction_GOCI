# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:56:29 2021
1. extract sub dataset for test
2. train_datda_analysis
3. earlystopping
4. plot_scatter
@author: Administrator
"""
import h5py
import numpy as np
import glob
import torch
from matplotlib import pyplot as plt
import torch
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde

def test_data(path):
    rate = 0.7
    data = h5py.File(path)
    x = np.array(data['train'])
    x[:,-3:] = np.cos(np.radians(x[:,-3:]))
    y = np.array(data['label'])
    z = np.array(data['cloud_Rrs'])
    # x_mean = np.mean(x,axis=0)
    # x_std = np.std(x,axis=0)
    # y_mean = np.mean(y,axis=0)
    # y_std = np.std(y,axis=0)
    # print(x_mean,'\n',x_std,'\n',y_mean,'\n',y_std)
    
    length = len(x)
    induces = torch.randperm(length)
    xx = x[induces[:int(length*rate)],:]
    yy = y[induces[:int(length*rate)],:]
    zz = z[induces[:int(length*rate)],:]
    xxx = x[induces[int(length*rate):],:]
    yyy = y[induces[int(length*rate):],:]
    zzz = z[induces[int(length*rate):],:]
    print('train mean:',np.mean(x,axis=0))
    print('train std:',np.std(x,axis=0))
    print('label mean:',np.mean(y,axis=0))
    print('label std:',np.std(y,axis=0))
    
    f = h5py.File('train_ds.h5','w')
    f.create_dataset('train',data=xx)
    f.create_dataset('label',data=yy)
    f.create_dataset('cloud_Rrs',data=zz)
    f.close()
    f1 = h5py.File('test_ds.h5','w')
    f1.create_dataset('train',data=xxx)
    f1.create_dataset('label',data=yyy)
    f1.create_dataset('cloud_Rrs',data=zzz)
    f1.close()
    
def OCI2(band1,band2,band3):
    '''
    MODIS
    band1:443
    band2:555
    band3:667
    '''
    ci = band2-0.5*(band1+band3)
    ci[np.argwhere(ci>0.004)] = 0.004   #ci指数小于-0.0005的可用于计算chl, 大概范围【-0.008，0.004】
    # chl = 10**(-0.4909+191.6590*ci)   #OCI 1
    chl = 10**(-0.4287+230.47*ci)       #OCI 2
    return chl.squeeze()
    
def extract_chl_OCI2(path):
    '''根据叶绿素浓度提取数据集'''
    thre = 200000
    data = h5py.File(path)
    x = np.array(data['train'])
    y = np.array(data['label'])
    chl = OCI2(y[:,1],y[:,5],y[:,6])
    
    levels = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35]
    train = np.zeros([1,17])
    label = np.zeros([1,8])
    for i,level in enumerate(levels):
        print(str(i+1),'/',str(len(levels)))
        up = level+0.05
        low = level
        idx = np.array(list(set(np.argwhere(chl>low).flatten()).intersection(set(np.argwhere(chl<up).flatten()))))
        if len(idx)>thre:
            #随机取thre个数据
            rand_num = torch.randperm(len(idx)).numpy()
            train = np.concatenate((train,x[idx[rand_num[:thre]],:]),axis=0)
            label = np.concatenate((label,y[idx[rand_num[:thre]],:]),axis=0)
        else:
            print('The number is not enough at:',level)
    train = np.delete(train,0,0)
    label = np.delete(label,0,0)
    f = h5py.File('train_data_OCI2.h5','w')
    f.create_dataset('train',data=train)
    f.create_dataset('label',data=label)
    f.close()
    
def extract_turbid(path):
    '''根据667nm提取数据集'''
    thre = 200000
    data = h5py.File(path)
    x = np.array(data['train'])
    y = np.array(data['label'])
    turbid = y[:,-2]
    
    levels = [0.0000,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007]
    train = np.zeros([1,17])
    label = np.zeros([1,8])
    for i,level in enumerate(levels):
        print(str(i+1),'/',str(len(levels)))
        up = level+0.0001
        low = level
        idx = np.array(list(set(np.argwhere(turbid>low).flatten()).intersection(set(np.argwhere(turbid<up).flatten()))))
        if len(idx)>thre:
            #随机取thre个数据
            rand_num = torch.randperm(len(idx)).numpy()
            train = np.concatenate((train,x[idx[rand_num[:thre]],:]),axis=0)
            label = np.concatenate((label,y[idx[rand_num[:thre]],:]),axis=0)
        else:
            train = np.concatenate((train,x[idx,:]),axis=0)
            label = np.concatenate((label,y[idx,:]),axis=0)
            print('The number is not enough at:',level,len(idx))
    train = np.delete(train,0,0)
    label = np.delete(label,0,0)
    f = h5py.File('train_data_667nm.h5','w')
    f.create_dataset('train',data=train)
    f.create_dataset('label',data=label)
    f.close()
def train_data_analysis(path):
    
    files = glob.glob(path+'dataset_*.h5')
    i = 0
    for file in files:
        i+=1
        print('part:'+str(i))
        data = h5py.File(file)
        x = np.array(data['train'])
        y = np.array(data['label'])
        if i == 1:
            #metrics calculating
            mean_rrc = np.mean(x,axis=0)[np.newaxis,:] #including geometries
            std_rrc = np.std(x,axis=0)[np.newaxis,:]
            min_rrc = np.amin(x,axis=0)[np.newaxis,:]
            max_rrc = np.amax(x,axis=0)[np.newaxis,:]
            
            mean_rrs = np.mean(y,axis=0)[np.newaxis,:] #including geometries
            std_rrs = np.std(y,axis=0)[np.newaxis,:]
            min_rrs = np.amin(y,axis=0)[np.newaxis,:]
            max_rrs = np.amax(y,axis=0)[np.newaxis,:]
            
            watertype = x[:,-1]
            types = np.unique(watertype)
            types = types[:,np.newaxis]
            for j,k in enumerate(types[:,0]):
                types[j,1] = np.sum(watertype==k)
            print(types)
        else:
            mean_rrc1 = np.mean(x,axis=0)[np.newaxis,:] #including geometries
            std_rrc1 = np.std(x,axis=0)[np.newaxis,:]
            min_rrc1 = np.amin(x,axis=0)[np.newaxis,:]
            max_rrc1 = np.amax(x,axis=0) [np.newaxis,:]
            mean_rrs1 = np.mean(y,axis=0)[np.newaxis,:] #including geometries
            std_rrs1 = np.std(y,axis=0)[np.newaxis,:]
            min_rrs1 = np.amin(y,axis=0)[np.newaxis,:]
            max_rrs1 = np.amax(y,axis=0)[np.newaxis,:]
            
            mean_rrc = np.concatenate((mean_rrc[:],mean_rrc1),axis=0)
            std_rrc = np.concatenate((std_rrc,std_rrc1),axis=0)
            min_rrc = np.concatenate((min_rrc,min_rrc1),axis=0)
            max_rrc = np.concatenate((max_rrc,max_rrc1),axis=0)
            mean_rrs = np.concatenate((mean_rrs,max_rrs1),axis=0) 
            std_rrs = np.concatenate((std_rrs,std_rrs1),axis=0)
            min_rrs = np.concatenate((min_rrs,min_rrs1),axis=0)
            max_rrs = np.concatenate((max_rrs,max_rrs1),axis=0)
            
    mean_rrc = np.mean(mean_rrc,axis=0) #including geometries
    std_rrc = np.mean(std_rrc,axis=0)
    min_rrc = np.amin(min_rrc,axis=0)
    max_rrc = np.amax(max_rrc,axis=0)
    mean_rrs = np.mean(mean_rrs,axis=0) #including geometries
    std_rrs = np.mean(std_rrs,axis=0)
    min_rrs = np.amin(min_rrs,axis=0)
    max_rrs = np.amax(max_rrs,axis=0)
            
    return mean_rrc,std_rrc,min_rrc,max_rrc,mean_rrs,std_rrs,min_rrs,max_rrs
     
class EarlyStopping:
    #https://blog.csdn.net/qq_37430422/article/details/103638681
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7,verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, 'best_model.pt')	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

def plot_scatter(x1,y1,stats_dl,band):
    '''
    x: predict
    y: label
    '''
    min_val = np.amin([x1,y1])
    max_val = np.amax([x1,y1])
    
    x0 = np.array([min_val,max_val])
    y0 = np.array([min_val,max_val])
    
    # ZX = (mat1-np.mean(mat1))/np.std(mat1)
    # ZY = (mat2-np.mean(mat2))/np.std(mat2)
    # r = np.sum(ZX*ZY)/(len(mat1))
    r2_1 = r2_score(x1,y1)
    rpd_1 = np.mean((x1-y1)/y1)
    bias_1 = np.mean(y1-x1)
    
    #density scatter
    xy1 = np.vstack([x1, y1])
    z = gaussian_kde(xy1)(xy1)
    idx = z.argsort()
    x1, y1, z1 = x1[idx], y1[idx], z[idx]
    
    #创建绘图
    fig = plt.figure(num=1, figsize=(3.5, 3.0))    #figsize单位为英寸
    ax = plt.subplot(111)
    # 设置字体
    # plt.rcParams['axes.unicode_minus'] = False          #使用上标小标小一字号
    # plt.rcParams['font.sans-serif']=['Times New Roman'] #设置全局字体，‘SimHei’黑体可现实中文
    font1 = { 
         'color': 'black',
         'weight': 'normal', #wight为字体的粗细，可选 ‘normal\bold\light’等
         'size': 10
         }
    font2 = {
         'color': 'black',
         'weight': 'normal', 
         'size': 13
        }
    font3 = { 
         'color': 'darkorange',
         'weight': 'normal', #wight为字体的粗细，可选 ‘normal\bold\light’等
         'size': 5
         }
    # plt.rc('font', family='Times New Roman', size=7)    
    #设置x,y轴的风格
    # ax.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True,
    #                direction='inout', labelsize=10, width=1, length=10, colors='black')
    # ax.tick_params(axis='x', which='minor', bottom=True, top=False, labelbottom=True,
    #                direction='in', labelsize=10, width=1, length=5,colors='black')
    # ax.tick_params(axis='y', which='major', left=True, right=False,labelbottom=True,
    #                direction='inout', labelsize=10, width=1, length=10,colors='black')
    # ax.tick_params(axis='y', which='minor', left=True, right=False,labelbottom=True,
    #                direction='in', labelsize=10, width=1, length=5,colors='black')
#        f1 = ax.plot(x, y, marker='o', markersize=1.2, color='blue', linewidth=0.0, linestyle='--')
    plt.xlim(min_val,max_val)
    plt.ylim(min_val,max_val)
    #绘图
    f0 = ax.plot(x0,y0, color='black',linewidth=1)
    f1 = ax.scatter(x1,y1, marker='o', s=3, c=z1,cmap='rainbow',linewidth=0.0)    #x轴为insitu，y轴predict
    
    # plt.text(0.05,0.9, 'N=%d'%(len(x1)),fontdict=font1,transform=ax.transAxes)
    plt.text(0.05,0.9,'MAPD:%6.2f'%(stats_dl[0]*100)+'%',fontdict=font1,transform=ax.transAxes)
    plt.text(0.05,0.8, 'R:%6.2f'%(stats_dl[1]),fontdict=font1,transform=ax.transAxes)
    plt.text(0.05,0.7, 'bias:%8.4f sr$^{-1}$'%(stats_dl[2]),fontdict=font1,transform=ax.transAxes)

    # plt.title(name+band)
    # ax.tick_params(axis='y', direction='in', length=3, width=1, colors='black', labelrotation=90)

    #设置坐标名
    ax.set_ylabel(r'Retrieved R$\mathregular{_{rs}}$'+r' (sr$\mathregular{^{-1}}$)', fontdict=font2)
    ax.set_xlabel(r'R$\mathregular{_{rs,true}}$'+r' (sr$\mathregular{^{-1}}$)', fontdict=font2)
    # plt.minorticks_on()     #开启小坐标
    # ax.xaxis.set_label_coords(0.5, -0.11)
    # ax.tick_params(axis='x', direction='in', length=3, width=1, colors='black', labelrotation=0)
    # plt.ticklabel_format(axis='both',style='sci')   #sci文章的风格
    plt.tight_layout(rect=(0,0,1,1))#rect=[left,bottom,right,top]   #设置图框与图片边缘的距离
    figname = r'./GOCI_'+str(band)+'_500dpi.png'
    plt.savefig(figname, dpi=600)

    plt.close()
    
    
if __name__=='__main__':
     path = r'GOCI_ds_0623.h5'
     # mean_rrc,std_rrc,min_rrc,max_rrc,mean_rrs,std_rrs,min_rrs,max_rrs = train_data_analysis(path)
     # print(mean_rrc,std_rrc,min_rrc,max_rrc,mean_rrs,std_rrs,min_rrs,max_rrs)
     test_data(path)
    
    
    