
import numpy as np
import numpy.matlib as npm

def QAscores_6Bands(test_Rrs, test_lambda):
    '''Quality assurance system for Rrs spectra (Version 2.0)'''

    # function [maxCos, cos, clusterID, totScore] = QAscores_5Bands(test_Rrs, test_lambda)
    # 
    # In this version, five-band Rrs at 412, 443, 488, 551 and 670 nm are used
    # in order to create comparable QA scores for different instruments (e.g., SeawiFS, MODISA, MEIRS, VIIRS, OLCI).
    #
    # ------------------------------------------------------------------------------
    # KNOWN VARIABLES :   ref_nRrs   -- Normalized Rrs spectra per-determined from water clustering (23x5 matrix)  
    #                     ref_lambda -- Wavelengths for ref_nRrs (1x5 matrix)
    #                     upB        -- Upper boundary (23x5 matrix)
    #                     lowB       -- Lower boundary (23x5 matrix)
    #
    # INPUTS:            test_Rrs -- matrix (inRow*inCol), each row represents one Rrs spectrum; numpy array
    #                    test_lambda-- Wavelengths for test_Rrs; numpy array
    #
    # OUTPUTS:  maxCos     -- maximum cosine values
    #           cos        -- cosine values for every ref_nRrs spectra
    #           clusterID  -- idenfification of water types (from 1-23)
    #           totScore   -- total score assigned to test_Rrs
    # ------------------------------------------------------------------------------
    # 
    # NOTE:
    #         1) Five wavelengths (412, 443, 488, 551, 670 nm) are assumed in the model
    #         2) If your Rrs data were measured at other wavelength, e.g. 440nm, you may want to change 440 to 443 before the model run;
    #             or modify the code below to find a cloest wavelength from the nine bands.
    #         3) The latest version may be found online at HTTP://oceanoptics.umb.edu/score_metric
    #
    # Reference:
    #         Wei, Lee, and Shang (2016). 
    #         A system to measure the data quality of spectral remote sensing
    #         reflectance of aquatic environments. Journal of Geophysical Research, 
    #         121, doi:10.1002/2016JC012126
    #         
    # ------------------------------------------------------------------------------
    # Note:
    #     1) nanmean, nansum need statistics toolbox
    #     2) on less memory and multi-core system, it may further speedup using
    #        parfor
    #
    # Author: Jianwei Wei, NOAA/NESDIS Center for Satellite Applications and Research
    # Email: Jianwei.Wei@noaa.gov
    #
    # upated January-14-2020
    # 
    # Translated from the original Matlab to python by Dirk Aurin, 2020-11-25
    # ------------------------------------------------------------------------------

    ''' Check input data '''
    if test_lambda.ndim > 1:
        row_lam, len_lam = test_lambda.shape
        if row_lam != 1:
            test_lambda = np.transpose(test_lambda)#np.transpose 矩阵转置
            row_lam, len_lam = test_lambda.shape
    else:
        row_lam = 1
        len_lam = len(test_lambda)

    row, col = test_Rrs.shape
    if len_lam != col and len_lam != row:
        print('Rrs and lambda size mismatch, please check the input data!')
    elif len_lam == row:
        test_Rrs = np.transpose(test_Rrs)

    ''' 23 Normalized spectral water types ''' 
    ref_lambda = np.array([412,443,490,555,660,680])
 
    ref_nRrs = np.array([\
        [0.73797, 0.53538, 0.33492, 0.07218, 0.00727, 0.00704],
        [0.67702, 0.53388, 0.39394, 0.10354, 0.01093, 0.01046],
        [0.60833, 0.52121, 0.43584, 0.14033, 0.01639, 0.01665],
        [0.50964, 0.47792, 0.46164, 0.20609, 0.02873, 0.03144],
        [0.42965, 0.43557, 0.47152, 0.25272, 0.03786, 0.04085],
        [0.36334, 0.38706, 0.45816, 0.3044 , 0.04205, 0.04688],
        [0.30946, 0.35492, 0.45121, 0.33479, 0.04777, 0.05227],
        [0.27593, 0.3148 , 0.41545, 0.37826, 0.06195, 0.06749],
        [0.34894, 0.33506, 0.39142, 0.37751, 0.0903 , 0.11766],
        [0.22773, 0.2753 , 0.38287, 0.42035, 0.07897, 0.08228],
        [0.29144, 0.2761 , 0.34217, 0.43707, 0.12861, 0.18141],
        [0.18747, 0.24076, 0.34199, 0.46108, 0.14677, 0.15051],
        [0.17256, 0.22029, 0.34231, 0.4639 , 0.09281, 0.09574],
        [0.18842, 0.2345 , 0.31897, 0.46281, 0.21459, 0.21402],
        [0.14302, 0.19142, 0.30576, 0.49178, 0.16956, 0.17984],
        [0.18122, 0.20035, 0.26124, 0.43693, 0.35886, 0.37375],
        [0.17377, 0.20335, 0.2826 , 0.4724 , 0.27161, 0.28031],
        [0.14173, 0.16884, 0.27937, 0.52527, 0.12058, 0.13119],
        [0.04976, 0.12646, 0.21885, 0.42293, 0.45169, 0.44941],
        [0.11665, 0.15256, 0.25801, 0.51451, 0.24308, 0.25949],
        [0.163  , 0.17546, 0.24907, 0.54423, 0.18972, 0.21692],
        [0.11145, 0.1349 , 0.22644, 0.51088, 0.30961, 0.32933],
        [0.14503, 0.13257, 0.1755 , 0.54786, 0.34124, 0.4489 ]])

    
    upB = np.array([\
        [0.7797 , 0.55909, 0.36692, 0.0959 , 0.0457 , 0.04662],
        [0.71136, 0.55484, 0.42432, 0.12594, 0.02795, 0.02748],
        [0.64637, 0.54024, 0.47083, 0.17318, 0.06701, 0.06176],
        [0.56956, 0.51481, 0.52763, 0.23993, 0.06184, 0.0616 ],
        [0.47766, 0.48771, 0.54753, 0.30075, 0.09892, 0.09822],
        [0.42349, 0.41629, 0.50574, 0.34536, 0.0653 , 0.07093],
        [0.36203, 0.38604, 0.48546, 0.36027, 0.08976, 0.09648],
        [0.3281 , 0.34343, 0.46353, 0.41232, 0.0944 , 0.14021],
        [0.42856, 0.36912, 0.43403, 0.41035, 0.16615, 0.17529],
        [0.28325, 0.31755, 0.47085, 0.45236, 0.12817, 0.12541],
        [0.35991, 0.31914, 0.37308, 0.4772 , 0.16996, 0.28445],
        [0.25323, 0.28678, 0.37399, 0.50687, 0.18317, 0.18819],
        [0.235  , 0.25303, 0.39214, 0.48827, 0.12801, 0.13376],
        [0.26335, 0.26326, 0.34984, 0.50705, 0.26195, 0.27604],
        [0.20166, 0.21944, 0.33294, 0.52093, 0.20348, 0.22379],
        [0.22951, 0.22363, 0.29607, 0.46463, 0.39304, 0.41905],
        [0.23209, 0.24386, 0.31588, 0.50286, 0.30238, 0.3129 ],
        [0.20171, 0.20442, 0.30892, 0.56042, 0.16311, 0.16977],
        [0.06566, 0.1469 , 0.23551, 0.43943, 0.47897, 0.49313],
        [0.15924, 0.18448, 0.29638, 0.57076, 0.29045, 0.29316],
        [0.23468, 0.23695, 0.29291, 0.60506, 0.24064, 0.28576],
        [0.15917, 0.16716, 0.25081, 0.57295, 0.35104, 0.38329],
        [0.18025, 0.16669, 0.19758, 0.57836, 0.37903, 0.50856]])
    
    lowB=np.array([\
        [0.70944, 0.51166, 0.27132, 0.04442, 0.00232, 0.00173],
        [0.6384 , 0.50884, 0.36351, 0.08433, 0.00275, 0.00301],
        [0.55333, 0.49738, 0.4116 , 0.11926, 0.00716, 0.00678],
        [0.43575, 0.43822, 0.41917, 0.16873, 0.01034, 0.01058],
        [0.36483, 0.39039, 0.41721, 0.20248, 0.01587, 0.01529],
        [0.30705, 0.3602 , 0.405  , 0.27164, 0.02872, 0.0281 ],
        [0.25106, 0.3148 , 0.41504, 0.3065 , 0.01641, 0.02128],
        [0.19524, 0.26646, 0.3746 , 0.34537, 0.02347, 0.02525],
        [0.29511, 0.31603, 0.36698, 0.34139, 0.05834, 0.06608],
        [0.13128, 0.23383, 0.33563, 0.37636, 0.02155, 0.03245],
        [0.24706, 0.24041, 0.31095, 0.37735, 0.08493, 0.11754],
        [0.14758, 0.20726, 0.30161, 0.4266 , 0.10952, 0.11484],
        [0.09174, 0.16101, 0.31322, 0.4364 , 0.02426, 0.02348],
        [0.15838, 0.19961, 0.26513, 0.43813, 0.15437, 0.17919],
        [0.06575, 0.14873, 0.2733 , 0.46632, 0.13489, 0.14272],
        [0.15597, 0.16064, 0.22583, 0.41662, 0.32762, 0.33207],
        [0.13659, 0.1762 , 0.25185, 0.43664, 0.2441 , 0.24336],
        [0.05794, 0.11578, 0.24911, 0.49879, 0.04967, 0.05421],
        [0.03211, 0.07956, 0.1825 , 0.411  , 0.41683, 0.40914],
        [0.03579, 0.09634, 0.21754, 0.49047, 0.2044 , 0.21684],
        [0.10724, 0.14053, 0.1988 , 0.50762, 0.14872, 0.17132],
        [0.07319, 0.09803, 0.20015, 0.48461, 0.26383, 0.29162],
        [0.0932 , 0.0945 , 0.14641, 0.48517, 0.30136, 0.38304]])

    refRow, _ = ref_nRrs.shape

    ''' Match the ref_lambda and test_lambda '''
    idx0 = np.empty(len(ref_lambda), dtype='int') # for ref_lambda 
    idx1 = np.empty(len(test_lambda), dtype='int') # for test_lambda

    for i, value in enumerate(test_lambda):
        pos, = np.where(ref_lambda == value) #find(ref_lambda==value);
        # if isempty(pos)
        if pos.size > 0:
            idx0[i] = pos.astype(int)
            idx1[i] = i
        else:
            idx1[i] = np.nan
            
    pos = np.isnan(idx1)
    np.delete(idx1, pos)

    test_lambda = test_lambda[idx1]
    test_Rrs = test_Rrs[:,idx1]
    ref_lambda = ref_lambda[idx0]
    ref_nRrs = ref_nRrs[:,idx0]
    upB = upB[:,idx0]
    lowB = lowB[:,idx0] 

    ''' Match the ref_nRrs and test_Rrs '''
    # keep the original value
    test_Rrs_orig = test_Rrs
    
    ''' Normalization '''
    inRow, inCol = np.shape(test_Rrs)

    # transform spectrum to column, inCol*inRow
    test_Rrs = np.transpose(test_Rrs)
    test_Rrs_orig = np.transpose(test_Rrs_orig)

    # inCol*inRow
    nRrs_denom = np.sqrt(np.nansum(test_Rrs**2, 0))
    # nRrs_denom = repmat(nRrs_denom,[inCol,1]);
    nRrs_denom = npm.repmat(nRrs_denom, inCol, 1)
    nRrs = test_Rrs/nRrs_denom;      

    # SAM input, inCol*inRow*refRow 
    test_Rrs2 = np.repeat(test_Rrs_orig[:, :, np.newaxis], refRow, axis=2)

    # #for ref Rrs, inCol*refRow*inRow 
    # test_Rrs2p = np.moveaxis(test_Rrs2, 2, 1)

    # inCol*inRow*refRow  
    nRrs2_denom = np.sqrt(np.nansum(test_Rrs2**2, 0))
    # nRrs2_denom = repeat(nRrs2_denom, inCol, axis=2)
    nRrs2_denom = np.repeat(nRrs2_denom[:,:, np.newaxis], inCol, axis=2)
    nRrs2_denom = np.moveaxis(nRrs2_denom, 2, 0)
    nRrs2 = test_Rrs2/nRrs2_denom
    # inCol*refRow*inRow  
    nRrs2 = np.moveaxis(nRrs2, 2, 1)

    ''' Adjust the ref_nRrs, according to the matched wavebands '''
    #row,_  = ref_nRrs.shape

    #### re-normalize the ref_adjusted
    ref_nRrs = np.transpose(ref_nRrs)

    # inCol*refRow*inRow 
    ref_nRrs2 = np.repeat(ref_nRrs[:,:, np.newaxis], inRow, axis=2)

    # inCol*refRow*inRow 
    ref_nRrs2_denom = np.sqrt(np.nansum(ref_nRrs2**2, 0))
    ref_nRrs2_denom = np.repeat(ref_nRrs2_denom[:,:, np.newaxis], inCol, axis=2)
    ref_nRrs2_denom = np.moveaxis(ref_nRrs2_denom, 2, 0)
    ref_nRrs_corr2 = ref_nRrs2/ref_nRrs2_denom

    ''' Classification '''
    #### calculate the Spectral angle mapper
    # inCol*refRow*inRow 
    cos_denom = np.sqrt(np.nansum(ref_nRrs_corr2**2, 0) * np.nansum(nRrs2**2, 0))
    cos_denom = np.repeat(cos_denom[:, :, np.newaxis], inCol, axis=2)
    cos_denom = np.moveaxis(cos_denom, 2, 0)
    cos = (ref_nRrs_corr2*nRrs2)/cos_denom
    # refRow*inRow 
    cos = np.sum(cos, 0)
    
    # 1*inRow
    maxCos = np.amax(cos, axis=0) 
    clusterID = np.argmax(cos, axis=0) # finds location of max along an axis, returns int64
    posClusterID = np.isnan(maxCos)

    ''' Scoring '''
    upB_corr = np.transpose(upB) 
    lowB_corr = np.transpose(lowB)

    ''' Comparison '''
    # inCol*inRow
    upB_corr2 = upB_corr[:,clusterID] * (1+0.01)
    lowB_corr2 = lowB_corr[:,clusterID] * (1-0.01)
    ref_nRrs2 = ref_nRrs[:,clusterID]

    #normalization
    ref_nRrs2_denom = np.sqrt(np.nansum(ref_nRrs2**2, 0))
    ref_nRrs2_denom = np.transpose(np.repeat(ref_nRrs2_denom[:,np.newaxis], inCol, axis=1))
    upB_corr2 = upB_corr2 / ref_nRrs2_denom
    lowB_corr2 = lowB_corr2 / ref_nRrs2_denom

    upB_diff = upB_corr2 - nRrs
    lowB_diff = nRrs - lowB_corr2

    C = np.empty([inCol,inRow], dtype='float')*0
    pos = np.logical_and(upB_diff>=0, lowB_diff>=0)
    C[pos] = 1

    #process all NaN spectral 
    C[:,posClusterID] = np.nan                                               

    totScore = np.nanmean(C, 0)
    clusterID = clusterID.astype('float')
    clusterID[posClusterID] = np.nan
    # Convert from index to water type 1-23
    clusterID = clusterID +1

    return maxCos, cos, clusterID, totScore