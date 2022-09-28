# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:28:42 2021
cloud detection algorithm
1 Wang and Shi, two NIR bands (Wang and Shi,IEEE TGRS, 2006)
2 Nordkvist method (Nordkvis, Opt.Express, 2009)
3 he shuangyan (Shiming Lu, RS, 2021)
@author: Administrator
"""
import numpy as np

def cloud_wang(band1,band2):
    '''
    Parameters, rayleigh correction reflectance
    ----------
    band1 : 745nm
    band2 : 865nm
    '''
    [h,w] = band2.shape
    cloud_flag = np.zeros((h,w),'float32')
    '''thick cloud'''
    idx_thick = np.where(band2>=0.06)
    cloud_flag[idx_thick] = 1
    
    '''thin cloud'''
    idx_thin = np.where((band2>0.027)&(band2<0.06)&(band1/band2<=1.15))
    cloud_flag[idx_thin] = 1
    
    return cloud_flag

def cloud_ndt(band1,band2,band3,band4):
    '''
    
    '''
    

    
    