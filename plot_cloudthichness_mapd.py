# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:34:35 2022

@author: Administrator
"""
import numpy as np
import os
from L2_flags import L3_mask
import h5py
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

def plot_cloudthickness_effect(x1,y1,name):
    '''
    x: predict
    y: label
    '''
    min_val = np.amin([x1,y1])
    max_val = np.amax([x1,y1])
    
    x0 = np.array([min_val,max_val])
    y0 = np.array([min_val,max_val])

    #density scatter
    xy1 = np.vstack([x1, y1])
    z = gaussian_kde(xy1)(xy1)
    idx = z.argsort()
    x1, y1, z1 = x1[idx], y1[idx], z[idx]
    
    x1 = x1.reshape(-1, 1)
    y1 = y1.reshape(-1, 1)
    
    #linear fitting for clear waters
    x_clear = x1[x1.squeeze()<0.06]
    y_clear = y1[x1.squeeze()<0.06]
    z1 = z1[x1.squeeze()<0.06]
    
    reg_clear = LinearRegression().fit(x_clear, y_clear)
    x_cle = np.linspace(np.min(x_clear), np.max(x_clear), 50).reshape(-1, 1)
    y_cle = reg_clear.predict(x_cle)
    
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
    #设置x,y轴的风格
    # ax.set_xscale('log')
    ax.set_yscale('log')
    # plt.xlim(1.05*min_val,0.6*max_val)
    # plt.ylim(1.05*min_val,0.6*max_val)
    #绘图
    f0 = ax.plot(x_cle,y_cle, color='black',linewidth=1,label='Fitting line')
    f1 = ax.scatter(x_clear,y_clear, marker='o', s=3, c=z1,cmap='rainbow',linewidth=0.0)    #x轴为insitu，y轴predict
    
    plt.legend(loc='center right',frameon=False)
    mean = np.mean(y_clear)
    std = np.std(y_clear)
    
    plt.text(0.5,0.2, 'Mean:%6.2f'%(mean)+'%',transform=ax.transAxes)
    plt.text(0.5,0.1,'STD:%6.2f'%(std),transform=ax.transAxes)
    # plt.text(0.5,0.1, 'Slope:%6.2f'%(reg_clear.coef_),transform=ax.transAxes)

    #设置坐标名
    ax.set_ylabel(r'MAPD [%]',fontdict=font2) #|$\mathregular{R_{rs,cloud}}$(443)-$\mathregular{R_{rs,true}}$(443)|/$\mathregular{R_{rs,true}}$(443)
    ax.set_xlabel('$\mathregular{R_{rc}(865)}$',fontdict=font2)
    # plt.minorticks_on()     #开启小坐标
    # ax.xaxis.set_label_coords(0.5, -0.11)
    # ax.tick_params(axis='x', direction='in', length=3, width=1, colors='black', labelrotation=0)
    # plt.ticklabel_format(axis='both',style='sci')   #sci文章的风格
    plt.tight_layout(rect=(0,0,1,1))#rect=[left,bottom,right,top]   #设置图框与图片边缘的距离
    figname = r'./GOCI_cloud_thickness_'+name+'_600dpi.png'
    plt.savefig(figname, dpi=600)

    plt.close()
    
if __name__=='__main__':
    path = r'test_predict.h5'
    
    data = h5py.File(path,'r')
    
    cloud = np.array(data['cloud_Rrs'])
    nocloud = np.array(data['label'])
    predict = np.array(data['predict'])
    ray = np.array(data['train'])
    '''label 光谱图 '''
    # num = 1000
    # cloud_rrs = cloud[:num,:].transpose()
    # nocloud_rrs = nocloud[:num,:].transpose()
    # ray_r = ray[:num,:].transpose()
    
    #plot scatter
    # x = x[y!=-0.015534002]
    # y = y[y!=-0.015534002]
    band = 1
    sample_idx = np.random.randint(0,len(cloud),10000)
    x = nocloud[sample_idx,band]
    y1 = cloud[sample_idx,band]
    y2 = predict[sample_idx,band]
    z = ray[sample_idx,7]

    #随着云量的增加，对Rrs的影响差异,去除异常值
    diff1 = 100*abs(y1-x)/x
    diff2 = 100*abs(y2-x)/x
    # outlier_low = np.percentile(diff,1)
    # outlier_up = np.percentile(diff,99)
    # z = z[list(diff)>outlier_low]
    # diff = diff[list(diff)>outlier_low]
    # z = z[list(diff)<outlier_up]
    # diff = diff[list(diff)<outlier_up]
    plot_cloudthickness_effect(z,diff1,'cloud_Rrs')
    # plot_cloudthickness_effect(z,diff2,'predict_Rrs')