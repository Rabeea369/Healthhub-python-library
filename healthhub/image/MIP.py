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
import copy
from matplotlib import pyplot
from scipy import ndimage
import ipywidgets as ipyw
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir

def axial(scan):
    '''
    This function takes a 3D numpy array and returns the Axial Maximum Intensity Projetion
    Parameters:
        (int/float) (3D numpy array) scan : 3D array for which maximum intensity projection is requived  


    Returns:
        (int/float) (3D numpy array) scan : MIP of the scan


    '''
    assert len(scan.shape)==3,"The input scan should be 3-dimensional"
    image = np.amax(scan, axis = 0)
    return(image)
def sagittal_left(scan):
    '''
    This function takes a 3D numpy array and returns the Sagittal Maximum Intensity Projetion of the left half.
    Parameters:
        (int/float) (3D numpy array) scan : 3D array for which maximum intensity projection is requived  


    Returns:
        (int/float) (3D numpy array) scan : MIP of the scan


    '''
    assert len(scan.shape)==3,"The input scan should be 3-dimensional"
    scan = np.array(scan)
    image = np.amax(scan[:,:,round(scan.shape[2]/2):], axis = 2)
    return(image)
def sagittal_right(scan):
    '''
    This function takes a 3D numpy array and returns the Sagittal Maximum Intensity Projetion of the right half
    Parameters:
        (int/float) (3D numpy array) scan : 3D array for which maximum intensity projection is requived  


    Returns:
        (int/float) (3D numpy array) scan : MIP of the scan


    '''
    assert len(scan.shape)==3,"The input scan should be 3-dimensional"
    scan = np.array(scan)
    image = np.amax(scan[:,:,:round(scan.shape[2]/2)], axis = 2)
    return(image)
def sagittal(scan):
    '''
    This function takes a 3D numpy array and returns the Sagittal Maximum Intensity Projetion
    Parameters:
        (int/float) (3D numpy array) scan : 3D array for which maximum intensity projection is requived  


    Returns:
        (int/float) (3D numpy array) scan : MIP of the scan


    '''
    assert len(scan.shape)==3,"The input scan should be 3-dimensional"
    scan = np.array(scan)
    image = np.amax(scan, axis = 2)
    return(image)
def coronal(scan):
    '''
    This function takes a 3D numpy array and returns the Coronal Maximum Intensity Projetion
    Parameters:
        (int/float) (3D numpy array) scan : 3D array for which maximum intensity projection is requived  


    Returns:
        (int/float) (3D numpy array) scan : MIP of the scan


    '''
    assert len(scan.shape)==3,"The input scan should be 3-dimensional"
    scan = np.array(scan)
    image = np.amax(scan, axis = 1)
    return(image)

def mask_overlay(img, mask, mode = 'axial'):
    '''
    This function takes the 3D scan, its mask and the desired mode : axial, coronal or sagittal and overlays the mask 
    on the scan with a MIP in the provided mode direction.
    Parameters:
            (int/float) (3D numpy array) img  : the 3D scan on which the 3D mask needs to be overlayed 
            (int) (3D numpy array) mask : The maks which needs to be overlayed. The mask should be the same shape as the scan
            (string) mode : the direction of MIP of the overlayed mask. This can be 'axial', 
                            'sagittal' or 'coronal'. The default is 'axial'. 


        Returns:
            (int/foat) (2D numpy array) RGB_img : The MIP of the mask overayed on the scan



    '''
    assert len(img.shape)==3,"The input image should be 3-dimensional"
    assert mask.shape==img.shape,"The input image and mask should have same size"

    if mode == 'coronal':
        ax = 1
    elif mode =='axial':
        ax = 0
    else:
        ax = 2
    s = img
    s_mips = np.amax(s, axis= ax)
    mini = np.amin(s_mips)
    maxi = np.amax(s_mips)
    d = maxi - mini
    s_mips = (s_mips - mini)/d
    m = copy.copy(mask)
    m_mips = np.amax(m, axis= ax)
    con_ones = m_mips
    overlayed = copy.copy(m_mips)
    con_zeros = 1 - con_ones

    RGB_img = np.zeros([s_mips.shape[0],s_mips.shape[1],3],int)
    RGB_img[:,:,0] = s_mips*con_zeros*255 + overlayed*255
    RGB_img[:,:,2] = s_mips*con_zeros*255 
    RGB_img[:,:,1] = s_mips*con_zeros*255  

    return RGB_img

