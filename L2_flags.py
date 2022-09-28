import numpy as np

def L3_mask(flags, l2flags):
    
    # path = r"C:\Users\Administrator\Desktop\A2011001045500.L2_LAC_OC.nc"
    # l2flags = np.array(h5.File(path)["/geophysical_data/l2_flags"])
    size1 = l2flags.shape
    
    good_pixel = np.ones(size1,dtype='int32')
    m = np.array([2], dtype='int32')
    #l1flags_mask
    bad_pixel_index = np.bitwise_and(l2flags,m**flags[0])
    
    for f in range(len(flags)):
        dkt = np.bitwise_and(l2flags,m**flags[f])         #dkt: I dont know what this means
        bad_pixel_index = np.bitwise_or(bad_pixel_index,dkt)
        
    good_pixel[bad_pixel_index>0] = 0
    
    # print("After L2 mask, remaining：",np.sum(good_pixel))
    return good_pixel



