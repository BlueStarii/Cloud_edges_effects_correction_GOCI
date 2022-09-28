# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 22:12:57 2021
validation with insitu data
1. predict with DL model
2. calculate uncertainty and error
@author: Administrator
"""
import h5py
import numpy as np
import torch
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
# from tansformer import tnet
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import r2_score
import random
from tools import plot_scatter
from L2wei_QA import QAscores_6Bands
import matplotlib.pyplot as plt
# from neuralnetwork import Net
   
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

    metrics = np.zeros([5])
    uncertainty = np.std((mat2-mat1)/mat1)
    apd = np.mean((abs(mat2-mat1))/mat1)
    RMSE = np.sqrt(np.mean(np.square(mat2-mat1)))
    # r2 = r2_score(mat1,mat2)
    bias = np.mean(mat2-mat1)
    
    metrics[0] = uncertainty
    metrics[1] = apd
    metrics[2] = r
    metrics[3] = RMSE
    metrics[4] = bias
    return metrics
def plot_frequence(x,score_insitu,score_nir,score_ACDL):
    
    #创建绘图
    fig = plt.figure(num=1, figsize=(4, 3.5))    #figsize单位为英寸
    ax = plt.subplot(111)
    # 设置字体
    # plt.rcParams['axes.unicode_minus'] = False          #使用上标小标小一字号
    # plt.rcParams['font.sans-serif']=['Times New Roman'] #设置全局字体，‘SimHei’黑体可现实中文
    font1 = {
         'color': 'black',
         'size': 12
         }
    font2 = {
         'color': 'black',
         'size': 12
        }
    
    #设置x,y轴的风格
    # ax.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True,
    #                direction='inout', labelsize=10, width=1,length=10, colors='black')
    # ax.tick_params(axis='x', which='minor', bottom=True, top=False, labelbottom=True,
    #                direction='in', labelsize=10, width=1,length=5, colors='black')
    # ax.tick_params(axis='y', which='major', left=True, right=False,labelbottom=True,
    #                direction='inout', labelsize=10, width=1,length=10, colors='black')
    # ax.tick_params(axis='y', which='minor', left=True, right=False,labelbottom=True,
    #                direction='in', labelsize=10, width=1,length=5, colors='black')
    # plt.xticks(weight='bold')
    # plt.yticks(weight='bold')
    #绘图,带误差棒的折线图   
    f1 = ax.plot(x,score_insitu,marker='^', markersize=5,c='black',markerfacecolor='white',label='R$\mathregular{_{rs,cloud}}$')
    f2 = ax.plot(x,score_nir,marker='s',markersize=5,c='darkorange',label='R$\mathregular{_{rs,true}}$')
    f3 = ax.plot(x,score_ACDL,marker='o',markersize=5,c='forestgreen',label='R$\mathregular{_{rs,pre}}$')
    
    ax.legend(loc=4,prop={'size': 13},frameon=False)
    plt.xticks(range(9))
    ax.set_xticklabels(['1', '7/8', '6/8', '5/8', '4/8', '3/8', '2/8', '1/8', '0'])
    #设置坐标名
    ax.set_xlabel(r'QA scores', fontdict=font2)
    # ax.set_ylabel(r'Area (10$^{7}$ km$^{2}$)', fontdict=font2)
    ax.set_ylabel(u'# of pixels (%)', fontdict=font2)
    
    # plt.minorticks_on()     #开启小坐标
    # ax.xaxis.set_label_coords(0.5, -0.11)
    # ax.tick_params(axis='x', direction='in', length=3, width=1, colors='black', labelrotation=0)
    # plt.ticklabel_format(axis='both',style='sci')   #sci文章的风格
    plt.tight_layout(rect=(0,0,1,1))#rect=[left,bottom,right,top]   #设置图框与图片边缘的距离
    figname = r'./' + 'QA_scores_500dpi.png'
    plt.savefig(figname, dpi=500)
    plt.close()
    
if __name__ == '__main__':
    #路径
    path = r'test_predict.h5'
    data = h5py.File(path,'r')
    label = np.array(data['label'])
    predict = np.array(data['predict'])
    cloud_Rrs = np.array(data['cloud_Rrs'])
    
    '''#################QA###################'''
    total_score_sd = [] #seadas
    total_score_dl = [] #predict
    total_score_cd = [] #cloud
    length_td = len(label)
    num_batch = round(length_td/10)
    test_lambda = np.array([412,443,490,555,660,680])

    for batch in range(10):
        if batch<9:
            test_Rrs_sd = label[batch*num_batch:num_batch*(batch+1),:6]
            test_Rrs_dl = predict[batch*num_batch:num_batch*(batch+1),:6]
            test_Rrs_cd = cloud_Rrs[batch*num_batch:num_batch*(batch+1),:6]
            maxCos_sd, cos_sd, clusterID_sd, totScore_sd = QAscores_6Bands(test_Rrs_sd, test_lambda)
            maxCos_dl, cos_dl, clusterID_dl, totScore_dl = QAscores_6Bands(test_Rrs_dl, test_lambda)
            maxCos_cd, cos_cd, clusterID_cd, totScore_cd = QAscores_6Bands(test_Rrs_cd, test_lambda)
            total_score_sd = np.concatenate((total_score_sd,totScore_sd))
            total_score_dl = np.concatenate((total_score_dl,totScore_dl))
            total_score_cd = np.concatenate((total_score_cd,totScore_cd))
        else:
            test_Rrs_sd = label[batch*num_batch:,:6]
            test_Rrs_dl = predict[batch*num_batch:,:6]
            test_Rrs_cd = cloud_Rrs[batch*num_batch:,:6]
            maxCos_sd, cos_sd, clusterID_sd, totScore_sd = QAscores_6Bands(test_Rrs_sd, test_lambda)
            maxCos_dl, cos_dl, clusterID_dl, totScore_dl = QAscores_6Bands(test_Rrs_dl, test_lambda)
            maxCos_cd, cos_cd, clusterID_cd, totScore_cd = QAscores_6Bands(test_Rrs_cd, test_lambda)
            total_score_sd = np.concatenate((total_score_sd,totScore_sd))
            total_score_dl = np.concatenate((total_score_dl,totScore_dl))
            total_score_cd = np.concatenate((total_score_cd,totScore_cd))
    
    print(np.mean(total_score_cd),np.mean(total_score_sd),np.mean(total_score_dl))
    
    f = h5py.File('QAS_test_dataset.h5','w')
    f.create_dataset('QAS_sd',data=total_score_sd)
    f.create_dataset('QAS_dl',data=total_score_dl)
    f.create_dataset('QAS_cd',data=total_score_cd)
    f.close()
    '''
    plot QA scores for insitu NIR and ACDL
    '''
    # f = h5py.File('QAS_test_dataset.h5','r')
    # total_score_sd=np.array(f['QAS_sd'])
    # total_score_dl=np.array(f['QAS_dl'])
    # total_score_cd=np.array(f['QAS_cd'])
    
    s1 = []
    s2 = []
    s3 = []
    scores = [1, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0]
    x = np.arange(len(scores))
    for i in range(len(scores)):
        # frequence_sd.append(100*len([value for value in totScore_sd if value>=(min_value+i*interval) and value<((i+1)*interval+min_value)])/len(totScore_sd))
        s1.append(np.sum(total_score_cd>=scores[i])/len(total_score_cd))
        s2.append(np.sum(total_score_sd>=scores[i])/len(total_score_sd))
        s3.append(np.sum(total_score_dl>=scores[i])/len(total_score_dl))
        
    s1 = np.array(s1,'float32')*100
    s2 = np.array(s2,'float32')*100
    s3 = np.array(s3,'float32')*100
        
    plot_frequence(x,s1,s2,s3)
 
    # '''############clear water, 667nm<0.0005######'''
    # clear_idx = np.argwhere(insitu[:,-2]<0.0005).squeeze()
    # predict = predict[clear_idx,:]
    # insitu = insitu[clear_idx,:]
    # SeaDAS = SeaDAS[clear_idx,:]
    # clusterID = clusterID[clear_idx]
    '''###########################################'''
    # stats_dl = np.zeros([5,8])
    # stats_sd = np.zeros([5,8])
    # stats_bc = np.zeros([5,8])

    # r2 = np.ones([1,8])
    # rpd = np.zeros([1,8])
    # band_name = ['412nm','443nm','488nm','531nm','547nm','555nm','667nm','678nm']
    # for i in range(8)[:1]:
    #     nan_value = np.argwhere(insitu[:,i]==-999)
    #     a = np.delete(insitu[:,i],nan_value.squeeze(),axis=0)
    #     b = np.delete(predict[:,i],nan_value.squeeze(),axis=0)
    #     c = np.delete(SeaDAS[:,i],nan_value.squeeze(),axis=0)
    #     d = np.delete(train[:,i],nan_value.squeeze(),axis=0)
    #     stats_dl[:,i] = metric(a, b)
    #     stats_sd[:,i] = metric(a, c)
    #     stats_bc[:,i] = metric(b, c)
        
    #     plot_scatter(dataset,a*100,b*100,a*100,c*100,stats_dl[:,i],stats_sd[:,i],band_name[i])
        
    # f = h5py.File('unc_error_MOBY.h5','w')
    # f.create_dataset('stats_dl',data=stats_dl)
    # f.create_dataset('stats_sd',data=stats_sd)
    # f.close()
