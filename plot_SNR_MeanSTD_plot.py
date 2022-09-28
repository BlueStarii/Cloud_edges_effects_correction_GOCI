# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 09:42:27 2021
SNR绘图 mean/std
去除SNR数据中所有的‘inf’值
计算每个bin的频数
绘制SNR曲线

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py

def frequent(array):
    fq_mat = []
    interval = 10
    boundary = [10,20,30,40,50,60,70,80,90,100, 110, 120,\
                130, 140, 150, 160, 170, 180, 190] #,110,120,150,200,250,300,350,400,450,500,550,600
    for up_b in boundary:
        fq_mat.append(np.sum((array>up_b-interval)&(array<up_b)))
        
    return np.array(fq_mat)

def plot_SNR():
    path = r'D:\2-cloudremove\3-SNR_QA'
    files = glob.glob(path+'\SNR_comparison*.h5')

    band_name = ['412','443','490','555','660','680']
    
    for band in range(6):
        #创建绘图
        fig = plt.figure(num=1, figsize=(3.5, 3))    #figsize单位为英寸
        ax = plt.subplot(111)
        # 设置字体
        plt.rcParams['axes.unicode_minus'] = False          #使用上标小标小一字号
        # plt.rcParams['font.sans-serif']=['Times New Roman'] #设置全局字体，‘SimHei’黑体可现实中文
        font1 = {
             'color': 'black',
             'weight': 'normal', #wight为字体的粗细，可选 ‘normal\bold\light’等
             'size': 6
             }
        font2 = {
             'color': 'black',
             'weight': 'normal', 
             'size': 12
            }
        
        #设置x,y轴的风格
        ax.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True,
                       direction='inout', labelsize=10, length=10,width=1, colors='black')
        ax.tick_params(axis='x', which='minor', bottom=True, top=False, labelbottom=True,
                       direction='in', labelsize=10,length=5, width=1, colors='black')
        ax.tick_params(axis='y', which='major', left=True, right=False,labelbottom=True,
                       direction='inout', labelsize=10,length=10, width=1, colors='black')
        ax.tick_params(axis='y', which='minor', left=True, right=False,labelbottom=True,
                       direction='in', labelsize=10,length=5, width=1, colors='black')
        '''x轴'''
        x = [10,20,30,40,50,60,70,80,90,100, 110, 120,130, 140, 150, 160, 170, 180, 190]
        colorbar1 = ['darkblue','blue','slateblue','darkslateblue','blueviolet',\
                    'violet','fuchsia','mediumvioletred','deeppink','lavenderblush']
        colorbar2 = ['lightcoral','red','brown','indianred','lightcoral','salmon','tomato',\
                     'coral','sienna','chocolate']
        # style = ['-','--','-.','.','x','o','p','*','s','+']
        file_name = ['3×3','5×5','7×7','9×9','11×11','13×13','15×15','17×17','19×19','21×21']
        # labels = ['(a)','(b)','(c)','(d)','(e)','(f)']
        max_sd = []
        max_dl = []
        for i,file in enumerate(files):
            data = h5py.File(r'D:\2-cloudremove\3-SNR_QA\SNR_comparison'+str((i+1)*2+1)+'.h5','r')
            dl = np.array(data['deeplearning'+band_name[band]])
            sd = np.array(data['seadas'+band_name[band]])
            '''去除inf'''
            dl = dl[dl!=np.inf]
            sd = sd[sd!=np.inf]
            
            y_dl = dl
            y_sd = sd
            
            '''频数计算'''
            y_dl = frequent(y_dl)
            y_sd = frequent(y_sd)
            
            # max_sd.append(x[y_sd.index(max(y_sd))])
            # max_dl.append(x[y_dl.index(max(y_dl))])
            max_sd.append(np.sum(x*np.array(y_sd))/np.sum(y_sd))
            max_dl.append(np.sum(x*np.array(y_dl))/np.sum(y_dl))
            #绘图   
            f1 = ax.plot(x,y_sd/1e6,'-',markersize=2,linewidth=1.0,c=colorbar1[i],label=file_name[i])
            f2 = ax.plot(x,y_dl/1e6,'-',markersize=2,linewidth=1.0,c=colorbar2[i],label=file_name[i])
            # f3 = ax.plot(x3,y3*100, marker='s',markersize=2,linewidth=1.0,c='seagreen',label='Uncertainty_ACDL')
            # f4 = ax.plot(x4,y4*100, marker='s',markersize=2,linewidth=1.0,c='darkorange',label='Uncertainty_SeaDAS')
        mean_sd = np.mean(max_sd)
        mean_dl = np.mean(max_dl)
        plt.axvline(x=mean_sd, ls='--', c='blue') # 添加垂直线
        plt.axvline(x=mean_dl, ls='--', c='red') 
        # ax.set_yscale('log')
        # ax.legend()
        plt.text(0.4,0.4, 'Mean value:',fontdict=font2,transform=ax.transAxes)
        plt.text(0.4,0.3,'SeaDAS:%6.2f'%(mean_sd),fontdict={
                 'color': 'blue',
                 'weight': 'normal', 
                 'size': 12},transform=ax.transAxes)
        plt.text(0.4,0.2,'DLTC:%6.2f'%(mean_dl),fontdict={
                 'color': 'red',
                 'weight': 'normal', 
                 'size': 12},transform=ax.transAxes)
        # plt.text(0.8,0.9, labels[band],fontdict={
        #                                          'weight':'bold',
        #                                          'size':18},transform=ax.transAxes)
        #设置坐标名
        ax.set_ylabel(r'# number of pixels (×10$\mathregular{^6}$)', fontdict=font2)
        ax.set_xlabel(u'Signal/Noise', fontdict=font2)
        # ax2.set_ylabel(u'MAPD (%)', fontdict=font2)
        plt.minorticks_on()     #开启小坐标
        # ax.xaxis.set_label_coords(0.5, -0.11)
        # ax.tick_params(axis='x', direction='in', length=3, width=1, colors='black', labelrotation=0)
        # plt.ticklabel_format(axis='both',style='sci')   #sci文章的风格
        plt.tight_layout(rect=(0,0,1,1))#rect=[left,bottom,right,top]   #设置图框与图片边缘的距离
        # plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)
        figname = r'./' + 'SNR_'+band_name[band]+'_500dpi.png'
    
        # fig.legend(ncol=10,loc=9,prop={'size':15,'weight':'bold'},
                   # frameon=False,bbox_to_anchor=(0.5,1.05))
        plt.savefig(figname, dpi=500)
        plt.close()

if __name__=='__main__':    
    '''main function'''
    plot_SNR()
    