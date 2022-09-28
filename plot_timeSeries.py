# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 17:34:33 2021
SP/NA chl时间序列制图,上接 build_monthly_dt_NA.py
1、计算CHL
2、绘制时间序列曲线
@author: Administrator
"""
import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt

def CI(band1,band2,band3):
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

def plot_ts(chl_sd,chl_dl):
    #创建绘图
    fig = plt.figure(num=1, figsize=(5, 3))    #figsize单位为英寸
    ax = plt.subplot(111)
    # 设置字体
    plt.rcParams['axes.unicode_minus'] = False          #使用上标小标小一字号
    plt.rcParams['font.sans-serif']=['Times New Roman'] #设置全局字体，‘SimHei’黑体可现实中文
    font1 = {'family': 'Times New Roman', 
         'color': 'black',
         'weight': 'bold', #wight为字体的粗细，可选 ‘normal\bold\light’等
         'size': 10
         }
    font2 = {'family': 'Times New Roman', 
         'color': 'black',
         'weight': 'bold', 
         'size': 10
        }
    
    #设置x,y轴的风格
    ax.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True,
                   direction='inout', labelsize=10, width=1, colors='black')
    ax.tick_params(axis='x', which='minor', bottom=True, top=False, labelbottom=True,
                   direction='in', labelsize=10, width=1, colors='black')
    ax.tick_params(axis='y', which='major', left=True, right=False,labelbottom=True,
                   direction='inout', labelsize=10, width=1, colors='black')
    ax.tick_params(axis='y', which='minor', left=True, right=False,labelbottom=True,
                   direction='in', labelsize=10, width=1, colors='black')
    '''x轴'''
    x = ['January','February','March','April','May','June','July','August','September','October','November','December']
    
    #绘图,带误差棒的折线图   
    f1 = ax.errorbar(x,chl_sd[:,0],yerr=chl_sd[:,1],fmt='-o',markersize=5,linewidth=1.0,c='darkorange',label='SeaDAS')
    f2 = ax.errorbar(x,chl_dl[:,0],yerr=chl_dl[:,1],fmt='-s',markersize=5,linewidth=1.0,c='seagreen',label='ACDL')
    
    ax.legend()
    #设置坐标名
    ax.set_ylabel(r'Chl concentration (mg m$^{-3}$)', fontdict=font2)
    ax.set_xlabel(u'Month in 2005', fontdict=font2)
    plt.xticks(rotation=45)
    # ax2.set_ylabel(u'MAPD (%)', fontdict=font2)
    plt.minorticks_on()     #开启小坐标
    # ax.xaxis.set_label_coords(0.5, -0.11)
    # ax.tick_params(axis='x', direction='in', length=3, width=1, colors='black', labelrotation=0)
    # plt.ticklabel_format(axis='both',style='sci')   #sci文章的风格
    plt.tight_layout(rect=(0,0,1,1))#rect=[left,bottom,right,top]   #设置图框与图片边缘的距离
    figname = r'./' + 'CHL_TS_SP_500dpi.png'
    plt.savefig(figname, dpi=500)

    plt.close()

'''main function'''           
files = np.array(glob.glob(r'Z:\NA_2003_*_data.h5'))

CHL_sd = np.zeros((12,2),'float32')
CHL_dl = np.zeros((12,2),'float32')

'''读取数据'''
for i,file in enumerate(files):
    print('%d / %d'%(i,len(files)))
    data = h5py.File(file,'r')
    predict = np.array(data['ACDL'])
    label = np.array(data['seadas'])
    chlsd = CI(label[:,1],label[:,5],label[:,6])
    chldl = CI(predict[:,1],predict[:,5],predict[:,6])
    CHL_sd[i,0] = np.mean(chlsd)
    CHL_dl[i,0] = np.mean(chldl)
    CHL_sd[i,1] = np.std(chlsd)
    CHL_dl[i,1] = np.std(chldl)
    
f = h5py.File('CHL-std_timeseries_NA.h5','w')
f.create_dataset('chl_std_sd',data=CHL_sd)
f.create_dataset('chl_std_dl',data=CHL_dl)
f.close()
'''读取数据'''
f = h5py.File('CHL-std_timeseries_SP.h5','r')
CHL_sd = np.array(f['chl_std_sd'])
CHL_dl = np.array(f['chl_std_dl'])

'''plot'''
plot_ts(CHL_sd,CHL_dl)



