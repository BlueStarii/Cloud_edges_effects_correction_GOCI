# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:20:08 2021
global chl production composite for 2012.01 and 2012.07
save file as hdf with lat and lon layers
@author: Administrator
"""
import numpy as np
import h5py               #h5py 和 gdal不兼容
import glob
from osgeo import gdal,osr
import os

def write_bands(im_data, banddes=None):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_height, im_width, im_bands = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    # 数据类型必须有，因为要计算需要多大内存空间
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create("", im_width, im_height, im_bands, datatype)

    # 写入数组数据
    if im_bands == 1:
        # dataset.GetRasterBand(1).SetNoDataValue(65535)
        try:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入
        except:
            dataset.GetRasterBand(1).WriteArray(im_data[:,:,0])
    else:
        # if banddes==None:
        # banddes = ['Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_520', 'Rrs_565', 'Rrs_670', 'chlor_a']
        for i in range(im_bands):
            try:
                # dataset.GetRasterBand(i + 1).SetNoDataValue(65535)
                RasterBand = dataset.GetRasterBand(i + 1)
                # RasterBand.SetDescription(banddes[i])
                RasterBand.WriteArray(im_data[:, :, i])
            except IndentationError:
                print('band:'+i)

    return dataset

def array2tiff(raster,array,lat,lon):
    lonMin,latMax,lonMax,latMin = [lon.min(),lat.max(),lon.max(),lat.min()]
    N_lat = len(lat)  
    N_lon = len(lon)
    xsize = (lonMax - lonMin) / (float(N_lon)-1)
    ysize = (latMax - latMin) / (float(N_lat)-1)
    originX, originY = lonMin-(xsize/2),latMax+(ysize/2)
    cols = array.shape[1] #列数
    rows = array.shape[0] #行数
    
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(raster, cols, rows, 1, gdal.GDT_Float32)  #创建栅格
    outRaster.SetGeoTransform((originX, xsize, 0, originY, 0, -ysize))  #tiff范围
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)  #写入数据
    outRasterSRS = osr.SpatialReference()  #获取地理坐标系
    outRasterSRS.ImportFromEPSG(4326) # 定义输出的坐标系为"WGS 84"，AUTHORITY["EPSG","4326"]
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
def modis_gcp(longitude,latitude,inpt,outpt,label,radi_bands=['chlor_a'], latlon=False):
    if latlon:
        value = np.empty((longitude.shape[0], longitude.shape[1], len(radi_bands) + 2))
    else:
        value = np.empty((longitude.shape[0], longitude.shape[1], len(radi_bands)))
    try:
        # 如果需要加入经纬度数据
        # value[:, :, len(radi_bands)-1] = inpt
        # value[:, :, len(radi_bands)] = latitude
        # value[:, :, len(radi_bands) + 1] = longitude
        value = inpt
    except:
        pass        
    # 将波段数据写入临时内存文件
    image: gdal.Dataset = write_bands(value)
    # 控制点列表, 设置7*7个控制点
    gcps = []
    x_arr = np.linspace(0, longitude.shape[1] - 1, num=7, endpoint=True, dtype=np.int)
    y_arr = np.linspace(0, longitude.shape[0] - 1, num=7, endpoint=True, dtype=np.int)
    for x in x_arr:
        for y in y_arr:
            if abs(longitude[y, x]) > 180 or abs(latitude[y, x]) > 90:
                continue
            gcps.append(gdal.GCP(np.float64(longitude[y, x]), np.float64(latitude[y, x]),
                                 0,
                                 np.float64(x), np.float64(y)))
    # 设置空间参考
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    sr.MorphToESRI()
    # 给数据及设置控制点及空间参考
    image.SetGCPs(gcps, sr.ExportToWkt())

    outfile = os.path.dirname(outpt) + os.sep + os.path.basename(outpt) + label+'.tif'

    # 校正
    if latlon:
        cutlinelayer = radi_bands+['latitude', 'longitude']
    else:
        cutlinelayer = radi_bands
    dst = gdal.Warp(outfile, image, format='GTiff', tps=True, xRes=0.01, yRes=0.01, dstNodata=np.nan,
              resampleAlg=gdal.GRA_NearestNeighbour, cutlineLayer=cutlinelayer)  # dstNodata=65535

    for i,bandname in enumerate(cutlinelayer):
        band = dst.GetRasterBand(i+1)
        band.SetMetadata({'bandname':bandname})
        band.SetDescription(bandname)
    image: None
    return outfile

if __name__ == "__main__":

    path = r'D:\2-cloudremove\2-single_scene\G2019276'
    files = glob.glob(path+'\*G2019276031535_Rrs.h5')
    for file in files:
        '''image read'''
        geo_data_name = file[:-7]+'.L2_LAC_OC'
        geo_data = h5py.File(geo_data_name,'r')
        lat = np.array(geo_data['/navigation_data/latitude'])
        lon = np.array(geo_data['navigation_data/longitude'])
        
        data = h5py.File(file,'r')
        bands = ['412','443','490','555','660','680']
        [h,w] = lat.shape
        rpd = np.zeros([h,w])
        '''load qa score'''
        score_dl = np.array(data['score_dl'])
        score_sd = np.array(data['score_sd'])
        for band in range(len(bands)):
            Rrs_sd = np.array(data['seadas_'+str(band+1)])
            Rrs_dl = np.array(data['ACDL_'+str(band+1)])
            Rrs_sd[score_sd<1] = 0
            Rrs_dl[score_dl<1] = 0
            modis_gcp(lon,lat,Rrs_sd,file[:-7],bands[band]+'_sdqa',radi_bands=['chlor_a'])
            modis_gcp(lon,lat,Rrs_dl,file[:-7],bands[band]+'_dlqa',radi_bands=['chlor_a'])
            
            '''(Rrs_dl-Rrs_sd)/Rrs_sd'''
            rpd = (Rrs_dl-Rrs_sd)/Rrs_sd
            rpd[Rrs_sd<0] = 0
            rpd[Rrs_dl<0] = 0
            modis_gcp(lon,lat,rpd,file[:-7],bands[band]+'_rpdqa',radi_bands=['chlor_a'])

        
