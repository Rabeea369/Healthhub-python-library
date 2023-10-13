import numpy as np
import glob as glob
import SimpleITK as sitk
from tqdm import tqdm
import cv2
import pydicom
import os
import matplotlib.pyplot as plt
from PIL import Image
from tabulate import tabulate
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, feature
from scipy.spatial import distance
import skimage
from skimage import color, io
import math
from matplotlib import image
from matplotlib import pyplot
from scipy import ndimage
import ipywidgets as ipyw
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir


def tag(tags,meta_data,value):
    '''This function will take as input the tag, a list containing meta_data and the new value to assign to the tag and returns 
       a modified list of metadata with the updated tag value.
     Parameters:
            tags : a tag that identifies the attribute, usually in the format (XXXX,XXXX) with hexadecimal numbers
            meta_data :  list containing scan meta deta
            value : new value for the tag

    Returns:
            (list) ds : modified Meta data with new tag value 

    '''        
    ds = meta_data
    (x1,x2) = tags
    if (x1,x2) in ds:
        print('tag name: ',ds[x1,x2].keyword)
        print('previous value: ',ds[x1,x2].value)
        ds[x1,x2].value = value
        print('new value: ',ds[x1,x2].value)
    else:
        print('Tag not found')
    return ds

def image(image,output_path,dummy_dcm_path=''):
    '''
        This function saves the provided image (rgb or gray scale) to the defined directory. 
        It also takes in as input a path to a dcm file on which to save. This is optional.
    Parameters:
            image : numpy array
            output_path : string of the output directory where the image needs to be saved as a dcm file.
            dummy_dcm_path : string of directory of dcm file on which to write the image (optional)


    Returns:

    '''
    if dummy_dcm_path=='':
        dummy_dcm_path = 'dummy.dcm'

    ds = pydicom.dcmread(dummy_dcm_path)

    ds.Rows = image.shape[0]
    ds.Columns = image.shape[1]
    if len(image.shape)<2:
        imagee = copy.copy(image)
        imagee = np.dstack((imagee,imagee))
        image = np.dtack((imagee,image)) 
    image = image.astype(np.uint8)
    plt.imshow(image)
    ds.PhotometricInterpretation = 'RGB'
    ds.PixelRepresentation = 0
    ds.WindowCenter = 127.5
    ds.WindowWidth = 255.0
    ds.SamplesPerPixel = 3
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    if (0x0028, 0x0006) in ds:
        ds[0x0028, 0x0006].vaue = 0
    else:
        ds.add_new(0x00280006, 'US', 0)
    if (0x7fe0, 0x0010) in ds:
        ds.PixelData = image.tobytes()
    else:
        ds.add_new([0x7fe0, 0x0010], 'OB', image.tobytes())
    ds.save_as(output_path)
    return(ds)



def maxDiameter(full_mask,spacing):

    '''
    This function takes as input a 3D mask of object/nodule and the pre-defined pixel spacing (mm) in the 
    form (pixel_spcing_x,pixel_spacing_z) and finds the maximum diameter of the 3D object in physical units.
     Parameters:
            full_mask  : numpy array (dim_z, dim_y, dim_x) containing the 3D object mask
            spacing : (pixel_spcing_x , pixel_spacing_y , pixel_spacing_z) 


     Returns:
            (float) max_dist  : the length of the maximum diameter of 3D object
            (list)  max_pints : the circumference points of the diamter

    '''
    # Ful mask means that it should contain all the slices of nodule, so shape can be like [6,45,45]
    # spacing is array contains spacing of x,y and z axis, shape should be like [0.7,1.5]

#     full_mask = edge_mask(full_mask);    
    all_points = np.where(full_mask>0)
    x_coor = all_points[1]
    y_coor = all_points[2]
    z_coor = all_points[0]

    pairs = []
    coordinates = []
#     print('Number of points is {}'.format(z_coor.min()))

    for i in range(z_coor.shape[0]):
        coor = [int(x_coor[i]),int(y_coor[i]),int(z_coor[i])]
        coordinates.append(coor)

    cpy = coordinates[:]

    for p in coordinates:
        cpy.remove(p)
        for points in cpy:
            pr = [p,points]
            pairs.append(pr)

    distances = np.zeros(len(pairs))
    i = 0
#     print('Number of pairs is {}'.format(len(pairs)))
    for pair in pairs:
        p_1 = pair[0]
        p_2 = pair[1]
        distances[i] = math.sqrt(((p_1[0] - p_2[0])*spacing[0])**2 + ((p_1[1] - p_2[1])*spacing[1])**2 + ((p_1[2] - p_2[2])*spacing[2])**2)
        i = i+1
    max_dis = float(distances.max())
    max_p = np.where(distances==distances.max())[0]
    max_pints = []
    for p in max_p:
        max_pints.append(pairs[p])
    return max_dis, max_pints