# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 17:39:52 2021
绘制双Y轴图,对接SN_MdeianFilter.py
1 统计云厚度在[0.02,0.03,0.04,0.05,0.06,0.07]区间内的像元数量以及对应的MAPD均值
2 绘制双Y轴图（参考HU，2012，figure 12)
@author: Administrator
"""
import numpy as np
from matplotlib import pyplot as plt
import h5py

def count(data_nir,label,predict):
    '''统计云厚度区间内的N和索引'''
    interval = 0.02
    band_num = 6
    chl_ci = data_nir
    N = []
    chl_level = [0.02,0.03,0.04,0.05,0.06]
    MAPD = np.zeros((band_num,len(chl_level)))
    for i,level in enumerate(chl_level):
        low,up = level*(1-interval),level*(1+interval)
        idx_ = list(set(np.argwhere(chl_ci>low).flatten()).intersection(set(np.argwhere(chl_ci<up).flatten()))) #在叶绿素区间的index
        print('level '+str(level)+':'+str(len(idx_)))
        if len(idx_) != 0:
            N.append(len(idx_))
            for band in range(band_num):
                MAPD[band,i] = np.mean((predict[idx_,band]-label[idx_,band])/label[idx_,band])
                
        else:
            MAPD[:,i] = 0
            N.append(0)
    return MAPD,np.array(N,'float32')
            
            
path_rrs = 'test_predict.h5'
data = h5py.File(path_rrs,'r')
path_toa = 'test_ds.h5'
data_toa = h5py.File(path_toa,'r')
data_nir = np.array(data_toa['train'])[:,-4]
data_sd = np.array(data['label'])
data_dl = np.array(data['predict'])

chl_level = np.array([0.02,0.03,0.04,0.05,0.06],'float32')

MAPD,N = count(data_nir,data_sd,data_dl)

# f = h5py.File('stats_MAPD-N.h5','w')
# f.create_dataset('MAPD',data=MAPD)
# f.create_dataset('N',data=N)

# f = h5py.File('SN_MF_MAPD-N_NA.h5','r')
# MAPD_sd = np.array(f['MAPD_sd'])
# N_sd = np.array(f['N_sd'])
# MAPD_dl = np.array(f['MAPD_dl'])
# N_dl = np.array(f['N_dl'])

fig = plt.figure(num=1, figsize=(4, 3.0))    #figsize单位为英寸
ax = plt.subplot(111)
# 设置字体
plt.rcParams['axes.unicode_minus'] = False          #使用上标小标小一字号
plt.rcParams['font.sans-serif']=['Times New Roman'] #设置全局字体，‘SimHei’黑体可现实中文
font1 = {'family': 'Times New Roman', 
     'color': 'blue',
     'weight': 'bold', #wight为字体的粗细，可选 ‘normal\bold\light’等
     'size': 10
     }
font2 = {'family': 'Times New Roman', 
     'color': 'black',
     'weight': 'bold', 
     'size': 10
    }
#设置图例并且设置图例的字体及大小
font3 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 10,
}

# plt.rc('font', family='Times New Roman', size=7)    
#设置x,y轴的风格
ax.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True,
               direction='inout', labelsize=10, width=1, colors='black')
ax.tick_params(axis='x', which='minor', bottom=True, top=False, labelbottom=True,
               direction='in', labelsize=10, width=1, colors='black')
ax.tick_params(axis='y', which='major', left=True, right=False,labelbottom=True,
               direction='inout', labelsize=10, width=1, colors='black')
ax.tick_params(axis='y', which='minor', left=True, right=False,labelbottom=True,
               direction='in', labelsize=10, width=1, colors='black')

ax2 = ax.twinx()
ax2.tick_params(axis='y', which='major', left=False, right=True,labelbottom=True,
               direction='inout', labelsize=10, width=1, colors='blue')
ax2.tick_params(axis='y', which='minor', left=False, right=True,labelbottom=True,
               direction='in', labelsize=10, width=1, colors='blue')
ax2.spines['right'].set_color('blue')

#绘图   
# f1 = ax.plot(chl_level,MAPD[0,:]*100, c='black',label='412nm')
# f2 = ax.plot(chl_level,MAPD[1,:]*100, c='dimgray',label='443nm')
f3 = ax.plot(chl_level,MAPD[2,:]*100, c='grey',label='490nm')
f4 = ax.plot(chl_level,MAPD[3,:]*100, c='darkgrey',label='555nm')
f5 = ax.plot(chl_level,MAPD[4,:]*100, c='silver',label='660nm')
# f6 = ax.plot(chl_level,MAPD[5,:]*100, c='lightgrey',label='680nm')
f7 = ax2.plot(chl_level,N, c='blue',label='N')

'''set log axis'''
# ax.set_yscale('log')
# plt.xlim(10**-2,0.4)
# plt.ylim(-10**-1,10**6)
ax2.set_yscale('log')
# ax.ylim(0,10)

'''合并图例'''
lns = f3+f4+f5+f7
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=4,prop=font3)

#设置坐标名
ax.set_ylabel(r'MAPD (%)', fontdict=font2)
ax.set_xlabel(u'Cloud thickness (sr$^{-1}$)', fontdict=font2)
ax2.set_ylabel(r'# number of pixels', fontdict=font1)
# ax2.set_ylabel(u'MAPD (%)', fontdict=font2)
plt.minorticks_on()     #开启小坐标
# ax.xaxis.set_label_coords(0.5, -0.11)
# ax.tick_params(axis='x', direction='in', length=3, width=1, colors='black', labelrotation=0)
# plt.ticklabel_format(axis='both',style='sci')   #sci文章的风格
plt.tight_layout(rect=(0,0,1,1))#rect=[left,bottom,right,top]   #设置图框与图片边缘的距离
figname = r'./' + 'MAPD_N_500dpi.png'
plt.savefig(figname, dpi=500)

plt.close()



