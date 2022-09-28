# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:36:22 2022

@author: Administrator
"""
import numpy as np
import os
from L2_flags import L3_mask
import h5py
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

def cloud_iLu(band1,band2,band3,band4):
    '''
    improved Lu and He method 
    Parameters, rayleigh correction reflectance
    ----------
    band1:412 
    band2:660
    band3:680
    band4:865
    '''
    length = len(band2)
    cloud_flag = np.zeros((length,1),'int32')
    '''ratio'''
    ratio = np.maximum(np.maximum(band1,band2),np.maximum(band3,band4))/np.minimum(np.minimum(band1,band2),np.minimum(band3,band4))
    
    '''turbid water'''
    idx_turbid = np.argwhere((band1>=0.07)&(ratio<2.5)&(band4>=0.027))
    cloud_flag[idx_turbid,0] = 1
    
    '''clear water'''
    idx_clear = np.argwhere((band1/band2>=1)&(ratio<2.5)&(band4>=0.018))
    cloud_flag[idx_clear,0] = 1
    
    return cloud_flag


def plot_scatter(x,y,band):
    
    bands = ['412','443','490','555','660','680']
    xy1 = np.vstack([x, y])
    z = gaussian_kde(xy1)(xy1)
    idx = z.argsort()
    x1, y1, z1 = x[idx], y[idx], z[idx]
    
    min_val = np.amin([x,y])
    max_val = np.amax([x,y])
    
    x0 = np.array([min_val,max_val])
    y0 = np.array([0,0])
    y1 = np.array([min_val,max_val])
    
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    x_ = np.linspace(np.min(x), np.max(x), 50).reshape(-1, 1)
    y_ = reg.predict(x_)
    
    fig = plt.figure(num=1, figsize=(3.5, 3.0))    #figsize单位为英寸
    ax = plt.subplot(111)
    
    plt.xlim(min_val,0.8*max_val)
    plt.ylim(min_val,0.8*max_val)
    
    ZX = (x-np.mean(x))/np.std(x)
    ZY = (y-np.mean(y))/np.std(y)
    r = np.sum(ZX*ZY)/(len(x))
    mapd = np.mean(abs(y-x)/x)
    bias = np.mean(y-x)       
    
    plt.text(0.05,0.9, 'N=%d'%(len(x)),transform=ax.transAxes)
    plt.text(0.05,0.8,'MAPD:%6.2f'%(mapd),transform=ax.transAxes)
    plt.text(0.05,0.7, 'R:%6.2f'%(r),transform=ax.transAxes)
    plt.text(0.05,0.6, 'bias:%8.4f sr$^{-1}$'%(bias),transform=ax.transAxes)
    
    f1 = ax.scatter(x,y,c=z1,cmap='rainbow', s=1, marker='o')
    f2 = ax.plot(x0,y0, color='black',linewidth=1, linestyle='dashed')
    # f3 = ax.plot(x_, y_, color='r', linewidth=1., label='Fitting line')
    f4 = ax.plot(x0,y1,color='black',linewidth=1,label='1:1')
    ax.set_ylabel(r'$\mathregular{R_{rs,cloud}}$ $\mathregular{(sr^{-1})}$')
    ax.set_xlabel(r'$\mathregular{R_{rs,true}}$ $\mathregular{(sr^{-1})}$')
    # ax.legend()
    figname = r'./GOCI_cloud_effect1_'+bands[band]+'_500dpi.png'
    plt.ticklabel_format(axis='both',style='sci')   #sci文章的风格
    plt.tight_layout(rect=(0,0,1,1))#rect=[left,bottom,right,top]   #设置图框与图片边缘的距离
    plt.savefig(figname, dpi=600)

    plt.close()
    
def plot_scatter_old(x1,y1,band):
    '''
    x: predict
    y: label
    '''
    bands = ['412','443','490','555','660','680']
    min_val = np.amin([x1,y1])
    max_val = np.amax([x1,y1])
    
    x0 = np.array([min_val,max_val])
    y0 = np.array([min_val,max_val])

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
         'size': 14
        }
    font3 = { 
         'color': 'darkorange',
         'weight': 'normal', #wight为字体的粗细，可选 ‘normal\bold\light’等
         'size': 5
         }
    #设置x,y轴的风格

    plt.xlim(1.05*min_val,0.8*max_val)
    plt.ylim(1.05*min_val,0.8*max_val)
    #绘图
    f0 = ax.plot(x0,y0, color='black',linewidth=1)
    f1 = ax.scatter(x1,y1, marker='o', s=3, c=z1,cmap='rainbow',linewidth=0.0)    #x轴为insitu，y轴predict
    
    ZX = (x-np.mean(x))/np.std(x)
    ZY = (y-np.mean(y))/np.std(y)
    r = np.sum(ZX*ZY)/(len(x))
    mapd = np.mean(abs(y-x)/x)
    bias = np.mean(y-x)  
    # plt.text(0.05,0.9, 'N=%d'%(len(x)),weight='bold',fontsize=13,transform=ax.transAxes)
    plt.text(0.05,0.9, 'bias:%8.4f sr$^{-1}$'%(bias),weight='normal',fontsize=13,transform=ax.transAxes)
    plt.text(0.05,0.78,'MAPD:%6.2f'%(mapd),weight='normal',fontsize=13,transform=ax.transAxes)
    plt.text(0.05,0.66, 'R:%6.2f'%(r),weight='normal',fontsize=13,transform=ax.transAxes)
    
    
    # 修改坐标轴字体及大小
    # plt.yticks(fontproperties='Times New Roman', size=12,weight='bold')#设置大小及加粗
    # plt.xticks(fontproperties='Times New Roman', size=12,weight='bold')

    # plt.title(name+band)
    # ax.tick_params(axis='y', direction='in', length=3, width=1, colors='black', labelrotation=90)

    #设置坐标名
    ax.set_ylabel(r'$\mathregular{R_{rs,cloud}}$ $\mathregular{(sr^{-1})}$',fontdict=font2)
    ax.set_xlabel(r'$\mathregular{R_{rs,true}}$ $\mathregular{(sr^{-1})}$',fontdict=font2)
    # plt.minorticks_on()     #开启小坐标
    # ax.xaxis.set_label_coords(0.5, -0.11)
    # ax.tick_params(axis='x', direction='in', length=3, width=1, colors='black', labelrotation=0)
    # plt.ticklabel_format(axis='both',style='sci')   #sci文章的风格
    plt.tight_layout(rect=(0,0,1,1))#rect=[left,bottom,right,top]   #设置图框与图片边缘的距离
    figname = r'./GOCI_cloud_effect1_'+bands[band]+'_500dpi.png'
    plt.savefig(figname, dpi=600)

    plt.close()
    
    
if __name__=='__main__':
    path = r'GOCI_ds_iLu.h5'
    
    data = h5py.File(path,'r')
    
    cloud = np.array(data['cloud_Rrs'])
    nocloud = np.array(data['label'])
    ray = np.array(data['train'])
    '''label 光谱图 '''
    # num = 1000
    # cloud_rrs = cloud[:num,:].transpose()
    # nocloud_rrs = nocloud[:num,:].transpose()
    # ray_r = ray[:num,:].transpose()
    
    #plot scatter
    # x = x[y!=-0.015534002]
    # y = y[y!=-0.015534002]
    sample_idx = np.random.randint(0,len(cloud),10000)
    # z = ray[sample_idx,7]
    #有云和无云情况下，Rrs的差异
    # low = np.percentile(y,1)
    # up = np.percentile(y,99)
    # x = x[list(y)>low]
    # z = z[list(y)>low]
    # y = y[list(y)>low]
    # x = x[list(y)<up]
    # z = z[list(y)<up]
    # y = y[list(y)<up]
    for band in range(6):
        x = nocloud[sample_idx,band]
        y = cloud[sample_idx,band]
        plot_scatter_old(x,y,band)
    #随着云量的增加，对Rrs的影响差异,去除异常值
    # diff = abs(y-x)/x
    # outlier_low = np.percentile(diff,1)
    # outlier_up = np.percentile(diff,99)
    # z = z[list(diff)>outlier_low]
    # diff = diff[list(diff)>outlier_low]
    # z = z[list(diff)<outlier_up]
    # diff = diff[list(diff)<outlier_up]
    # plot_cloud_effect(z,diff)