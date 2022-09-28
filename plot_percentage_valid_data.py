# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 20:09:58 2022

@author: Administrator
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def plot_graph(xlabel,perfect_dl,perfect_sd,mean_dl,mean_sd):
    
    #创建绘图
    fig = plt.figure(num=1, figsize=(4, 3.5))    #figsize单位为英寸
    #第一张图
    ax = plt.subplot(111)
    # 设置字体
    # plt.rcParams['axes.unicode_minus'] = False          #使用上标小标小一字号
    # plt.rcParams['font.sans-serif']=['Times New Roman'] #设置全局字体，‘SimHei’黑体可现实中文
    font1 = {
         'color': 'black',
         'weight': 'normal', #wight为字体的粗细，可选 ‘normal\bold\light’等
         'size': 12
         }
    font2 = {
         'color': 'black',
         'weight': 'normal', 
         'size': 12
        }
    font3 = {
         'color': 'blue',
         'weight': 'normal', 
         'size': 12
        }
    
    #设置x,y轴的风格
    # ax.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True,
    #                direction='inout', labelsize=12, width=1,length=10, colors='black')
    # ax.tick_params(axis='x', which='minor', bottom=True, top=False, labelbottom=True,
    #                direction='in', labelsize=12, width=1,length=5, colors='black')
    # ax.tick_params(axis='y', which='major', left=True, right=False,labelbottom=True,
    #                direction='inout', labelsize=12, width=1,length=10, colors='black')
    # ax.tick_params(axis='y', which='minor', left=True, right=False,labelbottom=True,
    #                direction='in', labelsize=12, width=1,length=5, colors='black')
    # plt.minorticks_on()     #开启小坐标
    '''x轴'''
    x = np.arange(8)

    # plt.ylim(0.01,0.18)
    # plt.xticks(weight='bold')
    # plt.yticks(weight='bold')
    '''双Y轴'''
    ax2 = ax.twinx()
    ax2.tick_params(axis='y', which='major',colors='blue')
    # ax2.tick_params(axis='y',colors='blue')
    ax2.spines['right'].set_color('blue')
    # plt.yticks(weight='bold')
    #绘图,带误差棒的折线图   
    f1 = ax.plot(x,perfect_dl,marker='o', markersize=5,c='black',markerfacecolor='white',label='DLTCC')
    f2 = ax.plot(x,perfect_sd,marker='o',markersize=5,c='black',label='SeaDAS')
    f3 = ax2.plot(x,mean_dl,marker='o', markersize=5,c='blue',markerfacecolor='white',label='DLTCC')
    f4 = ax2.plot(x,mean_sd,marker='o', markersize=5,c='blue',label='SeaDAS')
    lns = f1+f2+f3+f4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs,frameon=False)

    # plt.ylim(0,10)
    #设置坐标名
    ax.set_ylabel(r'# of pixels (%)', fontdict=font2)
    # ax.set_ylabel(r'Area (10$\mathregular{^{6}}$ km$\mathregular{^{2}}$)', fontdict=font2)
    ax2.set_ylabel(r'Mean QA scores',c='blue',weight='normal',fontdict=font3)
    # ax.axes.xaxis.set_visible(False)    #x轴标签不可见

    plt.xticks(range(0,8,1))   
    # ax3.legend(ncol=2,prop={'size': 10,'weight':'bold'},frameon=False)
    ax.set_xticklabels(xlabel)
    for xtick in ax.get_xticklabels():
        xtick.set_rotation(45)
    
    ax.set_xlabel(r'Time', fontdict=font2)
    # plt.xticks(weight='bold')
    # plt.yticks(weight='bold')
    
    # plt.minorticks_on()     #开启小坐标
    # ax.xaxis.set_label_coords(0.5, -0.11)
    # ax.tick_params(axis='x', direction='in', length=3, width=1, colors='black', labelrotation=0)
    # plt.ticklabel_format(axis='both',style='sci')   #sci文章的风格
    plt.tight_layout(rect=(0,0,1,1))#rect=[left,bottom,right,top]   #设置图框与图片边缘的距离
    # figname = r'./' + 'CHL_TS_2003-2020_NA_500dpi.png'
    figname = r'./' + 'Dayseries_valid_data.png'
    plt.savefig(figname, dpi=500)

    plt.close()

if __name__=='__main__':
    path = r'D:\2-cloudremove\2-single_scene\G2019276'
    files = glob.glob(path+os.sep+'*Rrs.h5')
    perfect_dl = []
    perfect_sd = []
    mean_dl = []
    mean_sd = []
    
    for file in files:
        data = h5py.File(file,'r')
        score_dl = np.array(data['score_dl'])
        score_sd = np.array(data['score_sd'])
        total_dl = np.sum(score_dl>0)
        total_sd = np.sum(score_sd>0)
        
        perfect_dl.append(np.sum(score_dl==1)/total_dl)
        perfect_sd.append(np.sum(score_sd==1)/total_sd)
        
        mean_dl.append(np.mean(score_dl[score_dl>0]))
        mean_sd.append(np.mean(score_sd[score_sd>0]))
    x = ['8:55','9:55','10:55','11:55','12:55','13:55','14:55','15:55']
    
    plot_graph(x,np.array(perfect_dl)*100,np.array(perfect_sd)*100,mean_dl,mean_sd)
