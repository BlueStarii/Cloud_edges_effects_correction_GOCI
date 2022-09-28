'''
对比有薄云和无薄云下Rrs的差异

'''
import numpy as np
import h5py
import glob
from L2wei_QA import QAscores_6Bands
from L2_flags import L3_mask

def cloud_wang(band1,band2):
    '''
    Parameters, rayleigh correction reflectance
    ----------
    band1 : 745nm
    band2 : 865nm
    '''
    length = len(band2)
    cloud_flag = np.zeros((length,1),'int32')
    '''thick cloud'''
    idx_thick = np.argwhere(band2>=0.06)
    cloud_flag[idx_thick,0] = 1
    
    '''thin cloud'''
    idx_thin = np.argwhere((band2>0.020)&(band2<0.06)&(band1/band2<=1.15))
    cloud_flag[idx_thin,0] = 1
    
    return cloud_flag

def cloud_Lu(band1,band2,band3,band4):
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

if __name__=='__main__':
    path = 'H:\DATA\GOCI\L2_rhos'
    clock_11_list = glob.glob(path+'\G2019???03*.L2_LAC_OC')  # all images at 11:00
    clock_12_list = glob.glob(path+'\G2019???04*.L2_LAC_OC')  # all images at 12:00
    train_dataset = np.empty((1,18),'float32') 
    #########################################################################
    #                       build 11:00 dataset                             #
    #########################################################################
    n = 0
    for file in clock_11_list:
        n += 1
        print('%d/%d ................ %s'%(n,len(clock_11_list),file))
        #read L2 data
        L2_data = h5py.File(file,'r')
        l2_flags = L2_data['/geophysical_data/l2_flags']
        flags = [0,1,3,4,5,6,10,12,14,15,16,19,20,21,22,24,25]
        goodpixels = L3_mask(flags, l2_flags)
        val_index = np.argwhere(goodpixels.flatten()==1)    #good pixels index
        
        geod = ['Rrs_412','Rrs_443','Rrs_490','Rrs_555','Rrs_660','Rrs_680',\
                'rhos_412','rhos_443','rhos_490','rhos_555','rhos_660','rhos_680',\
                'rhos_745','rhos_865',"sena","senz","sola","solz"]                      #inputs of the model
        value = np.empty((l2_flags.shape[0]*l2_flags.shape[1], len(geod)),'float32')    #all channels of a image
        true_value = np.empty((l2_flags.shape[0]*l2_flags.shape[1], len(geod)),'float32')
        
        '''read 11 : 00 image'''
        for i in range(len(geod)):
            dataset_band = L2_data["/geophysical_data/" + geod[i]]
            value_band = dataset_band[:, :] * 1.
            # value_band[value_band == -32767.] = np.nan
            true_value[:,i] = value_band.flatten()
            try:
                gain = dataset_band.attrs["scale_factor"][0]
                offset = dataset_band.attrs["add_offset"][0]
            except:
                gain = 1
                offset = 0
            true_value[:,i] = true_value[:,i]*gain + offset                              #combination of valid pixels for training

        '''11:00 Rrs with no cloud using iLu method'''
        idx_nocld_11 = np.argwhere(cloud_Lu(true_value[:,-12],true_value[:,-8],true_value[:,-7], true_value[:,-5]).squeeze()==0)    
        try:    # search file at 8:00
            print('search for file at 8:00')
            clock_8 = h5py.File(file[:-15]+'0'+file[-14:],'r')
            flags_8 = clock_8['/geophysical_data/l2_flags']
            for i in range(len(geod)):
                dataset_band = clock_8["/geophysical_data/" + geod[i]]
                value_band = dataset_band[:, :] * 1.
                # value_band[value_band == -32767.] = np.nan
                value[:,i] = value_band.flatten()
                try:
                    gain = dataset_band.attrs["scale_factor"][0]
                    offset = dataset_band.attrs["add_offset"][0]
                except:
                    gain = 1
                    offset = 0
                value[:,i] = value[:,i]*gain + offset
            '''8:00 image with cloud'''
            cld_pix_8 = cloud_Lu(value[:,-12],value[:,-8],value[:,-7], value[:,-5])
            idx_8 = list(set(val_index.flatten()).intersection(set(np.argwhere(cld_pix_8==1).flatten())))
            
            '''combine 8:00 Rayleigh reflt with 11:00 Rrs'''
            idx_final_8 = list(set(idx_8).intersection(set(idx_nocld_11.flatten())))
            new_comb = np.concatenate((true_value[idx_final_8,:6],value[idx_final_8,6:]),axis=1)
            train_dataset = np.concatenate((train_dataset,new_comb),axis=0)
        except:
            print('L2 file for 8:00 is missed')
            
        try:    # search file at 9:00
            print('search for file at 9:00')
            clock_9 = h5py.File(file[:-15]+'1'+file[-14:],'r')
            flags_9 = clock_9['/geophysical_data/l2_flags']
            for i in range(len(geod)):
                dataset_band = clock_9["/geophysical_data/" + geod[i]]
                value_band = dataset_band[:, :] * 1.
                # value_band[value_band == -32767.] = np.nan
                value[:,i] = value_band.flatten()
                try:
                    gain = dataset_band.attrs["scale_factor"][0]
                    offset = dataset_band.attrs["add_offset"][0]
                except:
                    gain = 1
                    offset = 0
                value[:,i] = value[:,i]*gain + offset
            '''9:00 image with cloud'''
            cld_pix_9 = cloud_Lu(value[:,-12],value[:,-8],value[:,-7], value[:,-5])
            idx_9 = list(set(val_index.flatten()).intersection(set(np.argwhere(cld_pix_9==1).flatten())))
            
            '''combine 9:00 Rayleigh reflt with 11:00 Rrs'''
            idx_final_9 = list(set(idx_9).intersection(set(idx_nocld_11.flatten())))
            new_comb = np.concatenate((true_value[idx_final_9,:6],value[idx_final_9,6:]),axis=1)
            train_dataset = np.concatenate((train_dataset,new_comb),axis=0)
        except:
            print('L2 file for 9:00 is missed')
            
        try:    # search file at 10:00
            print('search for file at 10:00')
            clock_10 = h5py.File(file[:-15]+'2'+file[-14:],'r')
            flags_10 = clock_10['/geophysical_data/l2_flags']
            for i in range(len(geod)):
                dataset_band = clock_10["/geophysical_data/" + geod[i]]
                value_band = dataset_band[:, :] * 1.
                # value_band[value_band == -32767.] = np.nan
                value[:,i] = value_band.flatten()
                try:
                    gain = dataset_band.attrs["scale_factor"][0]
                    offset = dataset_band.attrs["add_offset"][0]
                except:
                    gain = 1
                    offset = 0
                value[:,i] = value[:,i]*gain + offset
            '''10:00 image with cloud'''
            cld_pix_10 = cloud_Lu(value[:,-12],value[:,-8],value[:,-7], value[:,-5])
            idx_10 = list(set(val_index.flatten()).intersection(set(np.argwhere(cld_pix_10==1).flatten())))
            
            '''combine 10:00 Rayleigh reflt with 11:00 Rrs'''
            idx_final_10 = list(set(idx_10).intersection(set(idx_nocld_11.flatten())))
            new_comb = np.concatenate((true_value[idx_final_10,:6],value[idx_final_10,6:]),axis=1)
            train_dataset = np.concatenate((train_dataset,new_comb),axis=0)
        except:
            print('L2 file for 10:00 is missed')
            
        try:    # search file at 12:00
            print('search for file at 12:00')
            clock_12 = h5py.File(file[:-15]+'4'+file[-14:],'r')
            flags_12 = clock_12['/geophysical_data/l2_flags']
            for i in range(len(geod)):
                dataset_band = clock_12["/geophysical_data/" + geod[i]]
                value_band = dataset_band[:, :] * 1.
                # value_band[value_band == -32767.] = np.nan
                value[:,i] = value_band.flatten()
                try:
                    gain = dataset_band.attrs["scale_factor"][0]
                    offset = dataset_band.attrs["add_offset"][0]
                except:
                    gain = 1
                    offset = 0
                value[:,i] = value[:,i]*gain + offset
            '''12:00 image with cloud'''
            cld_pix_12 = cloud_Lu(value[:,-12],value[:,-8],value[:,-7], value[:,-5])
            idx_12 = list(set(val_index.flatten()).intersection(set(np.argwhere(cld_pix_12==1).flatten())))

            '''combine 12:00 Rayleigh reflt with 11:00 Rrs'''
            idx_final_12 = list(set(idx_12).intersection(set(idx_nocld_11.flatten())))
            new_comb = np.concatenate((true_value[idx_final_12,:6],value[idx_final_12,6:]),axis=1)
            train_dataset = np.concatenate((train_dataset,new_comb),axis=0)
        except:
            print('L2 file for 12:00 is missed')
            
        ''' 删除小于0,大于1的光谱'''
        for band in range(14):
            idx_1 = np.argwhere(train_dataset[:,band]<=0)  #delete Rrs<=0
            train_dataset = np.delete(train_dataset,idx_1,axis=0)
        
            idx_2 = np.argwhere(train_dataset[:,band]>=1)  #delete Rrs>=1
            train_dataset = np.delete(train_dataset,idx_2,axis=0)
        print('--11:00 extract complete:--',file[-24:])
    #########################################################################
    #                       build 12:00 dataset                             #
    #########################################################################
    n = 0
    for file in clock_12_list:
        n += 1
        print('%d/%d ................ %s'%(n,len(clock_12_list),file))
        #read L2 data
        L2_data = h5py.File(file,'r')
        l2_flags = L2_data['/geophysical_data/l2_flags']
        flags = [0,1,3,4,5,6,10,12,14,15,16,19,20,21,22,24,25]
        goodpixels = L3_mask(flags, l2_flags)
        val_index = np.argwhere(goodpixels.flatten()==1)    #good pixels index
        
        geod = ['Rrs_412','Rrs_443','Rrs_490','Rrs_555','Rrs_660','Rrs_680',\
                'rhos_412','rhos_443','rhos_490','rhos_555','rhos_660','rhos_680',\
                'rhos_745','rhos_865',"sena","senz","sola","solz"]                      #inputs of the model
        value = np.empty((l2_flags.shape[0]*l2_flags.shape[1], len(geod)),'float32')    #all channels of a image
        true_value = np.empty((l2_flags.shape[0]*l2_flags.shape[1], len(geod)),'float32')
    
        '''read 12 : 00 image'''
        for i in range(len(geod)):
            dataset_band = L2_data["/geophysical_data/" + geod[i]]
            value_band = dataset_band[:, :] * 1.
            # value_band[value_band == -32767.] = np.nan
            true_value[:,i] = value_band.flatten()
            try:
                gain = dataset_band.attrs["scale_factor"][0]
                offset = dataset_band.attrs["add_offset"][0]
            except:
                gain = 1
                offset = 0
            true_value[:,i] = true_value[:,i]*gain + offset                              #combination of valid pixels for training
            
        '''12:00 Rrs with no cloud using Wang method'''
        idx_nocld_12 = np.argwhere(cloud_Lu(true_value[:,-12],true_value[:,-8],true_value[:,-7], true_value[:,-5]).squeeze()==0)
        try:    # search file at 11:00
            print('search for file at 11:00')
            clock_11 = h5py.File(file[:-15]+'3'+file[-14:],'r')
            flags_11 = clock_11['/geophysical_data/l2_flags']
            for i in range(len(geod)):
                dataset_band = clock_11["/geophysical_data/" + geod[i]]
                value_band = dataset_band[:, :] * 1.
                # value_band[value_band == -32767.] = np.nan
                value[:,i] = value_band.flatten()
                try:
                    gain = dataset_band.attrs["scale_factor"][0]
                    offset = dataset_band.attrs["add_offset"][0]
                except:
                    gain = 1
                    offset = 0
                value[:,i] = value[:,i]*gain + offset
            '''11:00 image with cloud'''
            cld_pix_11 = cloud_Lu(value[:,-12],value[:,-8],value[:,-7], value[:,-5])
            idx_11 = list(set(val_index.flatten()).intersection(set(np.argwhere(cld_pix_11==1).flatten())))
            
            '''combine 11:00 Rayleigh reflt with 11:00 Rrs'''
            idx_final_11 = list(set(idx_11).intersection(set(idx_nocld_12.flatten())))
            new_comb = np.concatenate((true_value[idx_final_11,:6],value[idx_final_11,6:]),axis=1)
            train_dataset = np.concatenate((train_dataset,new_comb),axis=0)
        except:
            print('L2 file for 11:00 is missed')
            
        try:    # search file at 13:00
            print('search for file at 13:00')
            clock_13 = h5py.File(file[:-15]+'5'+file[-14:],'r')
            flags_13 = clock_13['/geophysical_data/l2_flags']
            for i in range(len(geod)):
                dataset_band = clock_13["/geophysical_data/" + geod[i]]
                value_band = dataset_band[:, :] * 1.
                # value_band[value_band == -32767.] = np.nan
                value[:,i] = value_band.flatten()
                try:
                    gain = dataset_band.attrs["scale_factor"][0]
                    offset = dataset_band.attrs["add_offset"][0]
                except:
                    gain = 1
                    offset = 0
                value[:,i] = value[:,i]*gain + offset
            '''13:00 image with cloud'''
            cld_pix_13 = cloud_Lu(value[:,-12],value[:,-8],value[:,-7], value[:,-5])
            idx_13 = list(set(val_index.flatten()).intersection(set(np.argwhere(cld_pix_13==1).flatten())))
            
            '''combine 13:00 Rayleigh reflt with 12:00 Rrs'''
            idx_final_13 = list(set(idx_13).intersection(set(idx_nocld_12.flatten())))
            new_comb = np.concatenate((true_value[idx_final_13,:6],value[idx_final_13,6:]),axis=1)
            train_dataset = np.concatenate((train_dataset,new_comb),axis=0)
        except:
            print('L2 file for 13:00 is missed')
            
        try:    # search file at 14:00
            print('search for file at 14:00')
            clock_14 = h5py.File(file[:-15]+'2'+file[-14:],'r')
            flags_14 = clock_14['/geophysical_data/l2_flags']
            for i in range(len(geod)):
                dataset_band = clock_14["/geophysical_data/" + geod[i]]
                value_band = dataset_band[:, :] * 1.
                # value_band[value_band == -32767.] = np.nan
                value[:,i] = value_band.flatten()
                try:
                    gain = dataset_band.attrs["scale_factor"][0]
                    offset = dataset_band.attrs["add_offset"][0]
                except:
                    gain = 1
                    offset = 0
                value[:,i] = value[:,i]*gain + offset
            '''14:00 image with cloud'''
            cld_pix_14 = cloud_Lu(value[:,-12],value[:,-8],value[:,-7], value[:,-5])
            idx_14 = list(set(val_index.flatten()).intersection(set(np.argwhere(cld_pix_14==1).flatten())))
           
            '''combine 14:00 Rayleigh reflt with 12:00 Rrs'''
            idx_final_14 = list(set(idx_14).intersection(set(idx_nocld_12.flatten())))
            new_comb = np.concatenate((true_value[idx_final_14,:6],value[idx_final_14,6:]),axis=1)
            train_dataset = np.concatenate((train_dataset,new_comb),axis=0)
        except:
            print('L2 file for 14:00 is missed')
            
        try:    # search file at 15:00
            print('search for file at 15:00')
            clock_15 = h5py.File(file[:-15]+'4'+file[-14:],'r')
            flags_15 = clock_15['/geophysical_data/l2_flags'] 
            for i in range(len(geod)):
                dataset_band = clock_15["/geophysical_data/" + geod[i]]
                value_band = dataset_band[:, :] * 1.
                # value_band[value_band == -32767.] = np.nan
                value[:,i] = value_band.flatten()
                try:
                    gain = dataset_band.attrs["scale_factor"][0]
                    offset = dataset_band.attrs["add_offset"][0]
                except:
                    gain = 1
                    offset = 0
                value[:,i] = value[:,i]*gain + offset
            '''15:00 image with cloud'''
            cld_pix_15 = cloud_Lu(value[:,-12],value[:,-8],value[:,-7], value[:,-5])
            idx_15 = list(set(val_index.flatten()).intersection(set(np.argwhere(cld_pix_15==1).flatten())))
            
            '''combine 15:00 Rayleigh reflt with 12:00 Rrs'''
            idx_final_15 = list(set(idx_15).intersection(set(idx_nocld_12.flatten())))
            new_comb = np.concatenate((true_value[idx_final_15,:6],value[idx_final_15,6:]),axis=1)
            train_dataset = np.concatenate((train_dataset,new_comb),axis=0)
        except:
            print('L2 file for 15:00 is missed')
            
        ''' 删除小于0,大于1的光谱'''
        for band in range(14):
            idx_1 = np.argwhere(train_dataset[:,band]<=0)  #delete Rrs<=0
            train_dataset = np.delete(train_dataset,idx_1,axis=0)
        
            idx_2 = np.argwhere(train_dataset[:,band]>=1)  #delete Rrs>=1
            train_dataset = np.delete(train_dataset,idx_2,axis=0)
        
        print('--12:00 extract complete:--',file[-24:])
    ##########################################
    #                  QA                    #
    ##########################################
    train_dataset = np.delete(train_dataset,0,0)
    length_td = len(train_dataset)
    num_batch = round(length_td/10)
    test_lambda = np.array([412,443,488,555,667,678])
    hq_data = np.zeros((1,len(geod)),'float32')
    for batch in range(10):
        test_Rrs = train_dataset[batch*num_batch:num_batch*(batch+1),:6]
        maxCos, cos, clusterID, totScore = QAscores_6Bands(test_Rrs, test_lambda)
        #筛选QA评分等于1的Rrs
        hq_rrs_index = np.argwhere(totScore>0.9) #hq_rrs: means high quality Rrs
        hq_data = np.concatenate((hq_data,train_dataset[hq_rrs_index.squeeze(),:]),axis=0)
        print('hq_data num:',len(hq_data))
    hq_data = np.delete(hq_data,0,0)
    ##########################################
    #          calcualate RRA                #
    ##########################################
    train_data = hq_data[:,6:] #0-5：rrs, 6-13:rrc,14-17:geometries
    train_label = hq_data[:,:6]
    
    train_dataset = train_data[:,:-1]
    train_dataset[:,-3] = np.cos(train_data[:,-3])
    train_dataset[:,-2] = np.cos(train_data[:,-1])
    train_dataset[:,-1] = abs(train_data[:,-4]-train_data[:,-2])
    
    idx1 = np.argwhere(train_dataset[:,10]>180)
    column_ra = train_dataset[:,-1]
    column_ra[idx1.squeeze()] = 360-column_ra[idx1.squeeze()]
    train_dataset[:,-1] = column_ra
    train_dataset[:,-1]=np.cos(train_dataset[:,-1])
    print('data len:',len(train_dataset))
    
    ##########################################
    #          Save the dataset              #
    ##########################################
    file_name = "GOCI_train_iLu.h5"
    f = h5py.File(file_name, "w")
    f.create_dataset('train', data=train_dataset)
    f.create_dataset('label', data=train_label)
    f.close()
