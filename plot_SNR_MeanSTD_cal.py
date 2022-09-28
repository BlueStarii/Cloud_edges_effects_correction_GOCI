# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:16:06 2021

compare the SNR between DL and SeaDAS
SNR: MEAN value / standard deviation

@author: Administrator

"""
import numpy as np
import h5py
import glob
import os

def SNR_cal(image,box_size):
    [h,w] = np.shape(image)
    SNR = []
    n = int(box_size/2)   
    for i in range(n,h-n,box_size):
        for j in range(n,w-n,box_size):
            if np.sum(image[i-n:i+n+1,j-n:j+n+1])>0:
                SNR.append(np.mean(image[i-n:i+n+1,j-n:j+n+1][image[i-n:i+n+1,j-n:j+n+1].nonzero()])/np.std(image[i-n:i+n+1,j-n:j+n+1][image[i-n:i+n+1,j-n:j+n+1].nonzero()]))
    return SNR
  
if __name__=='__main__':          
          
    path = r'D:\2-cloudremove\2-single_scene\G2019276'
    files = glob.glob(os.path.join(path,'*_Rrs.h5'))
    box_sizes = [3,5,7,9,11,13,15,17,19,21]
    SNR_sd = np.zeros((1,6),'float32')
    SNR_dl = np.zeros((1,6),'float32')
    
    for box_size in box_sizes:
        print('^^^^^^^^^^bs='+str(box_size)+'^^^^^^^^^^^^')
        SNR_412_sd = []
        SNR_412_dl = []
        SNR_443_sd = []
        SNR_443_dl = []
        SNR_490_sd = []
        SNR_490_dl = []
        SNR_555_sd = []
        SNR_555_dl = []
        SNR_660_sd = []
        SNR_660_dl = []
        SNR_680_sd = []
        SNR_680_dl = []
        
        '''读取数据'''
        for file in files:
            data = h5py.File(file,'r')
            '''SNR'''
            for j in range(6):
                if j==0:
                    label_mat = np.array(SNR_cal(np.array(data['seadas_'+str(j+1)]),box_size))
                    predict_mat = np.array(SNR_cal(np.array(data['ACDL_'+str(j+1)]),box_size))
                    SNR_412_sd = np.concatenate((SNR_412_sd,label_mat))
                    SNR_412_dl = np.concatenate((SNR_412_dl,predict_mat))
                    
                elif j==1:
                    label_mat = np.array(SNR_cal(np.array(data['seadas_'+str(j+1)]),box_size))
                    predict_mat = np.array(SNR_cal(np.array(data['ACDL_'+str(j+1)]),box_size))
                    SNR_443_sd = np.concatenate((SNR_443_sd,label_mat))
                    SNR_443_dl = np.concatenate((SNR_443_dl,predict_mat))
                    
                elif j==2:
                    label_mat = np.array(SNR_cal(np.array(data['seadas_'+str(j+1)]),box_size))
                    predict_mat = np.array(SNR_cal(np.array(data['ACDL_'+str(j+1)]),box_size))
                    SNR_490_sd = np.concatenate((SNR_490_sd,label_mat))
                    SNR_490_dl = np.concatenate((SNR_490_dl,predict_mat))
                    
                elif j==3:
                    label_mat = np.array(SNR_cal(np.array(data['seadas_'+str(j+1)]),box_size))
                    predict_mat = np.array(SNR_cal(np.array(data['ACDL_'+str(j+1)]),box_size))
                    SNR_555_sd = np.concatenate((SNR_555_sd,label_mat))
                    SNR_555_dl = np.concatenate((SNR_555_dl,predict_mat))
                    
                elif j==4:
                    label_mat = np.array(SNR_cal(np.array(data['seadas_'+str(j+1)]),box_size))
                    predict_mat = np.array(SNR_cal(np.array(data['ACDL_'+str(j+1)]),box_size))
                    SNR_660_sd = np.concatenate((SNR_660_sd,label_mat))
                    SNR_660_dl = np.concatenate((SNR_660_dl,predict_mat))
                    
                elif j==5:
                    label_mat = np.array(SNR_cal(np.array(data['seadas_'+str(j+1)]),box_size))
                    predict_mat = np.array(SNR_cal(np.array(data['ACDL_'+str(j+1)]),box_size))
                    SNR_680_sd = np.concatenate((SNR_680_sd,label_mat))
                    SNR_680_dl = np.concatenate((SNR_680_dl,predict_mat))
                    
            # SNR_sd = np.concatenate((SNR_sd,label_mat),axis=0)
            # SNR_dl = np.concatenate((SNR_dl,predict_mat),axis=0)

        # SNR_sd = np.delete(SNR_sd,0,0)
        # SNR_dl = np.delete(SNR_dl,0,0)
        f = h5py.File('SNR_comparison'+str(box_size)+'.h5','w')
        f.create_dataset('seadas412',data=SNR_412_sd)
        f.create_dataset('seadas443',data=SNR_443_sd)
        f.create_dataset('seadas490',data=SNR_490_sd)
        f.create_dataset('seadas555',data=SNR_555_sd)
        f.create_dataset('seadas660',data=SNR_660_sd)
        f.create_dataset('seadas680',data=SNR_680_sd)
        f.create_dataset('deeplearning412',data=SNR_412_dl)
        f.create_dataset('deeplearning443',data=SNR_443_dl)
        f.create_dataset('deeplearning490',data=SNR_490_dl)
        f.create_dataset('deeplearning555',data=SNR_555_dl)
        f.create_dataset('deeplearning660',data=SNR_660_dl)
        f.create_dataset('deeplearning680',data=SNR_680_dl)        
        f.close()
    


