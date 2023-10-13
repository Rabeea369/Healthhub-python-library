#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import glob as glob
import SimpleITK as sitk
from tqdm import tqdm
import cv2
import pydicom
import os
import os
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir


# In[2]:


def scan(folder_path): ##Public func
    '''
    This function takes path of scan as input and returns scan array in 4D and its metadeta in a list. 
    The 4D array contains multiple 3D arrays corresponding to each series of the scan.
    The meta data will be returned in a list corresponding to each slice in the scan.


    Parameters:
            scan_path : path of given dicom scan (string)

    Returns:
            (float): Scan (numpy array)
                    : Meta Data (list)
    '''      


    files = __sorted_files(folder_path)
    f, myscan, mytags = __read_scan(folder_path,files)
    return np.array(myscan), mytags


def __sorted_files(folder): ##Private function
    ins=[]
    ori = []
    ser=[]
    f= os.listdir(folder)
    for name in tqdm(f):
        itkimage = sitk.ReadImage(os.path.join(folder,name))
        temp= float(itkimage.GetMetaData('0020|0013')) # instance number 
        temp1= itkimage.GetMetaData('0020|000e') #series instance UID 
        ins.append(int(temp))
        ser.append(temp1)
    series=np.unique(ser)

    files=[[x for sr,_,x in sorted(zip(ser,ins,f)) if sr==s] for s in series]
    return files
    
def __read_scan(folder_path,files): ##Private func
    axial  =  np.array([1., 0., 0., 0., 1., 0.])
    coronal = np.array([1., 0., 0., 0., 0., -1.])
    sagittal= np.array([0., 1., 0., 0., 0., -1.])
    myscan = []
    mytags = []
    f=[]
    for i in tqdm (range (len(files))):
        img=[]
        file=[]
        tags = []
        for filename in files[i]:
            ds = pydicom.dcmread(os.path.join(folder_path,filename))
            numpyImage = ds.pixel_array
            del ds[0x7fe0, 0x0010]
            img.append(numpyImage)
            file.append(filename)
            tags.append(ds)
        mytags.append(tags)
        myscan.append(np.array(img))
        f.append(file)
    return f, myscan, mytags

def Slice(slice_path): ##Public
    '''
    This function takes path of slice as input and returns slice array and its metadeta.

    Parameters:
            slice_path : path of given dicom file

    Returns:
            (int)(float): Slice 
                 (list) : Meta Data
    '''         
    ds = pydicom.dcmread(slice_path)
    numpyImage = ds.pixel_array
    return numpyImage, ds

def tag(tag_id,ds): ##Public
    '''This function takes the tag code and metadata as input and returns the value of provided tag code.

    Parameters:
            tag_id : a tag that identifies the attribute, usually in the format (XXXX,XXXX) with hexadecimal numbers
            ds : meta deta of dicom slice

    Returns:
            (int)(float)(String): Tag Value 

    '''
    (x1,x2) = tag_id 
    if (x1,x2) in ds:
        print(ds[x1,x2])
        return ds[x1,x2].value
    else:
        print('Tag not found')
        return ''
        
def getSeriesPlane(ds): ##public
    '''This function takes the metadata as input and returns the tags associated with the series, 
       the orienation of the series and the seriest instance UID.

    Parameters:
            ds= meta deta

    Returns:
            Series Orientation, Series Instance UID                            
    '''
    l=[]
    if ((0x0020,0x0037) in ds):
        o=ds[0x0020,0x0037]

    else:
        o=''
    if ((0x0020,0x000e) in ds):
        s=ds[0x0020,0x000e]
        #print("hi")
    else:
        s=''
    l.append([o,s])

    return l

def orientation(orientation_tag_value): ##Public func
    '''
   This function takes the orientation tag value of a dicom file which is an array of floats and returns the associated 
   orientation such as axial, sagittal or coronal.

    Parameters:
            orientation_tag_value : orientation tag

    Returns:
            (string): Orientation of tag [axial, sagittal or coronal] 

    '''
    axial  =  np.array([1., 0., 0., 0., 1., 0.])
    coronal = np.array([1., 0., 0., 0., 0., -1.])
    sagittal= np.array([0., 1., 0., 0., 0., -1.])

    a = np.around(orientation_tag_value)

    if (a==coronal).all():
        return 'coronal'
    elif (a==axial).all():
        return 'axial'
    elif (a==sagittal).all():
        return 'sagittal'
    else:
        print('orientation not found')
        return ''

def seriesData(ds):   ##Public func
    '''This function returns an array od size [n,3] where n is the number of scans and of sopseriesUID, series_number and plane corresponding to each series.
     Parameters:
            ds  :  a list containing scan meta data

    Returns:
            (list):  [sopseriesUID, series_number, view corresponding to each series] 

    '''

    n=len(ds)
    l=[]
    for i in range(n):
        a = ds[i][0][0x0020,0x000e].value if ((0x0020,0x000e) in ds[i][0]) else ''
        if ((0x0020,0x0037) in ds[i][0]):
            b = self.orientation(ds[i][0][0x0020,0x0037].value)
        else:
            b = 'no orientation tag'
        c = ds[i][0][0x0020,0x0011].value if ((0x0020,0x0011) in ds[i][0]) else ''
        l.append(np.array([a,b,c]))

    return np.array(l)

def getImagePlane(meta_data):  ##Public func
    #Pixel Spacing
    '''This function takes the metadata as input and  returns  all the tags within ImagePlane  module
     Parameters:
            ds  :  a list containing scan meta deta

    Returns:
            (list): [Tag Name, Tag Code and Tag Values with in imagePlane Module] 

    '''
    l=[]
    ds = meta_data
    #a = ds[0x0020,0x000e].value
    if ((0x0020,0x0050) in ds):
        l.append(ds[0x0020,0x0050])
    if ((0x0020,0x0032) in ds):
        l.append(ds[0x0020,0x0032])
    if ((0x0020,0x0037) in ds):
        l.append(ds[0x0020,0x0037])
    if ((0x0020,0x1041) in ds):
        l.append(ds[0x0020,0x1041])
    if ((0x0020,0x0030) in ds):
        l.append(ds[0x0020,0x0030])

   # print(data)
    #l.append([a,a1,a2,a3,a4])
    return l

def getFrameOfReference(meta_data): ##Public func
    '''This function takes the metadata as input and  returns  all the tags within FrameOfReference  module
     Parameters:
            ds  :  a list containing scan meta deta

     Returns:
            (list) : [Tag Name, Tag Code and Tag Values with in FrameOfReference Module ]

     '''
    #data = meta_data.group_dataset(0x0020)
    l=[]
    ds = meta_data
    #a = ds[0x0020,0x000e].value
    if ((0x0020,0x0052) in ds):
        l.append(ds[0x0020,0x0052])

    if ((0x0020,0x1040) in ds):
        l.append(ds[0x0020,0x1040])

   # print(data)
    if l == []:
        print('No frame of reference tags found')
    return l

def getPatientStudy(meta_data):
    '''This function takes the metadata as input and  returns  all the tags within getPatientStudy  module
     Parameters:
            ds  : a list containing scan meta deta

     Returns:
            (list) : [Tag Name, Tag Code and Tag Values with in getPatientStudy Module] 

    '''  
    ds = meta_data
    l=[]
    #a = ds[0x0020,0x000e].value
    if ((0x0008,0x1080) in ds):
        l.append(ds[0x0008,0x1080])
    if ((0x0008,0x1084) in ds):
        l.append(ds[0x0008,0x1084])
    if ((0x0010,0x1010) in ds):
        l.append(ds[0x0010,0x1010])
    if ((0x0010,0x1020) in ds):
        l.append(ds[0x0010,0x1020])
    if ((0x0010,0x1021) in ds):
        l.append(ds[0x0010,0x1021])
    if ((0x0010,0x1022) in ds):
        l.append(ds[0x0010,0x1022])
    if ((0x0010,0x1023) in ds):
        l.append(ds[0x0010,0x1023])
    if ((0x0010,0x1024) in ds):
        l.append(ds[0x0010,0x1024])
    if ((0x0010,0x1030) in ds):
        l.append(ds[0x0010,0x1030])
    if ((0x0010,0x2000) in ds):
        l.append(ds[0x0010,0x2000])
    if ((0x0010,0x2110) in ds):
        l.append(ds[0x0010,0x2110])
    if ((0x0010,0x2180) in ds):
        l.append(ds[0x0010,0x2180])
    if ((0x0010,0x21A0) in ds):
        l.append(ds[0x0010,0x21A0])
    if ((0x0010,0x21B0) in ds):
        l.append(ds[0x0010,0x21B0])
    if ((0x0010,0x21D0) in ds):
        l.append(ds[0x0010,0x21D0])
    if ((0x0010,0x2203) in ds):
        l.append(ds[0x0010,0x2203])
    if ((0x0032,0x1066) in ds):
        l.append(ds[0x0032,0x1066])
    if ((0x0032,0x1067) in ds):
        l.append(ds[0x0032,0x1067])
    if ((0x0038,0x0010) in ds):
        l.append(ds[0x0038,0x0010])
    if ((0x0038,0x0014) in ds):
        l.append(ds[0x0038,0x0014])
    if ((0x0038,0x0060) in ds):
        l.append(ds[0x0038,0x0060])
    if ((0x0038,0x0062) in ds):
        l.append(ds[0x0038,0x0062])
    if ((0x0038,0x0064) in ds):
        l.append(ds[0x0038,0x0064])
    if ((0x0038,0x0500) in ds):
        l.append(ds[0x0038,0x0500])




    #l.append([a,a1,a2,a3,a4,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23]) 
    return l



    #col_names = ["Tag Name", "Tag Code","Tag Value"]

    #print(data)




def getGeneralStudy(meta_data): ##Public func
    '''This function takes the metadata as input and returns  all the tags within getGeneralStudy  module
     Parameters:
            ds  :  a list containing scan meta deta

     Returns:
            (list): [Tag Name, Tag Code and Tag Values with in getGeneralStudy Module] 

    '''   
    l=[]
    ds = meta_data
    #a = ds[0x0020,0x000e].value
    if ((0x0008,0x0020) in ds):
        l.append(ds[0x0008,0x0020])
    if ((0x0008,0x0030) in ds):
        l.append(ds[0x0008,0x0030])
    if ((0x0008,0x0050) in ds):
        l.append(ds[0x0008,0x0050])
    if ((0x0008,0x0051) in ds):
        l.append(ds[0x0008,0x0051])
    if ((0x0008,0x0090) in ds):
        l.append(ds[0x0008,0x0090])
    if ((0x0008,0x0096) in ds):
        l.append(ds[0x0008,0x0096])
    if ((0x0008,0x009C) in ds):
        l.append(ds[0x0010,0x009C])
    if ((0x0008,0x009D) in ds):
        l.append(ds[0x0010,0x009D])
    if ((0x0008,0x1030) in ds):
        l.append(ds[0x0008,0x1030])
    if ((0x0008,0x1032) in ds):
        l.append(ds[0x0008,0x1032])
    if ((0x0008,0x1048) in ds):
        l.append(s[0x0008,0x1048])
    if ((0x0008,0x1049) in ds):
        l.append(ds[0x0008,0x1049])
    if ((0x0008,0x1060) in ds):
        l.append(ds[0x0008,0x1060])
    if ((0x0008,0x1110) in ds):
        l.append(ds[0x0008,0x1110])
    if ((0x0020,0x000D) in ds):
        l.append(ds[0x0020,0x000D])
    if ((0x0020,0x0010) in ds):
        l.append(ds[0x0020,0x0010])
    if ((0x0032,0x1033) in ds):
        l.append(ds[0x0032,0x1033])
    if ((0x0032,0x01034) in ds):
        l.append(ds[0x0032,0x01034])
    if ((0x0040,0x1012) in ds):
        l.append(ds[0x0040,0x1012])





    #l.append([a,a1,a2,a3,a4,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18]) 
    return l


def getImagePixel(meta_data):    ##Public func
    '''This function takes a list containing the scan metadata as input and returns  all the tags within getImagePixel  module
     Parameters:
            ds  :  a list containing scan meta deta

     Returns:
            (list) : [Tag Name, Tag Code and Tag Values with in getImagePixel Module] 

    '''

    l=[]
    ds = meta_data
    #a = ds[0x0020,0x000e].value
    if ((0x0028,0x0002) in ds):
        l.append(ds[0x0028,0x0002])
    if ((0x0028,0x0004) in ds):
        l.append(ds[0x0028,0x0004])
    if ((0x0028,0x0006) in ds):
        l.append(ds[0x0028,0x0006])
    if ((0x0028,0x0010) in ds):
        l.append(ds[0x0028,0x0010])
    if ((0x0028,0x0010) in ds):
        l.append(ds[0x0028,0x0010])
    if ((0x0028,0x0011) in ds):
        l.append(ds[0x0028,0x0011])
    if ((0x0028,0x0034) in ds):
        l.append(ds[0x0028,0x0034])
    if ((0x0028,0x0100) in ds):
        l.append(ds[0x0028,0x0100])
    if ((0x0028,0x0101) in ds):
        l.append(ds[0x0028,0x0101])
    if ((0x0028,0x0102) in ds):
        l.append(ds[0x0028,0x0102])
    if ((0x0028,0x0103) in ds):
        l.append(ds[0x0028,0x0103])
    if ((0x0028,0x0106) in ds):
        l.append(ds[0x0028,0x0106])
    if ((0x0028,0x0107) in ds):
        l.append(ds[0x0028,0x0107])
    if ((0x0028,0x0121) in ds):
        l.append(ds[0x0008,0x0121])
    if ((0x0028,0x1101) in ds):
        l.append(ds[0x0028,0x01101])
    if ((0x0028,0x1102) in ds):
        l.append(ds[0x0028,0x1102])
    if ((0x0028,0x1103) in ds):
        l.append(ds[0x0028,0x1103])
    if ((0x0028,0x1201) in ds):
        l.append(ds[0x0028,0x1201])
    if ((0x0028,0x1202) in ds):
        l.append(ds[0x0028,0x1202])
    if ((0x0028,0x1203) in ds):
        l.append(ds[0x0028,0x1203])
    if ((0x0028,0x2000) in ds):
        l.append(ds[0x0028,0x2000])
    if ((0x0028,0x2002) in ds):
        l.append(ds[0x0028,0x2002])
    if ((0x0028,0x7FE0) in ds):
        l.append(ds[0x0028,0x7FE0])
    if ((0x0028,0x7FE0) in ds):
        l.append(ds[0x0028,0x7FE0])
    if ((0x7FE0,0x0001) in ds):
        l.append(ds[0x7FE0,0x0001])
    if ((0x7FE0,0x0002) in ds):
        l.append(ds[0x7FE0,0x0002])
    if ((0x7FE0,0x0010) in ds):
        l.append(ds[0x7FE0,0x0010])

    #l.append([a,a1,a2,a3,a4,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26]) 
    return l


# In[ ]:




