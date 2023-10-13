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

def volumeCalculator(slices,spacing):
    """
        This function takes 3D binary mask, of an object as an input along with pixel spacing and returns the physical volume as an output. The unit of volume returned 
        is same as the unit of pixel spacing (which is usually mm).

    Parameters:
        (int)(3D numpy ndarray) slices: 3D numpy array of Slices containing binary mask, usually of tumor. 
        (float)(list) spacing: a list of pixel spacing as [px,py,pz] where px,py and pz are 
                                pixel spacings in x,y and z coordinates. 

    Returns:
        (float) volume: The physical volume of the binary object in units same as pixel spacing units.
    """

    v=0

    if len(slices.shape) == 2:
        assert len(spacing) != 2, "Spacing should have two values, spacing should be a list of two numbers in case of 2D image"

        assert spacing[0] != None, "px cannot be None"
        assert spacing[1] != None, "py cannot be None"

        v = np.sum(slices)*spacing[0]*spacing[1]
    else:
        assert len(spacing) == 3, "pz should be defined"

        assert spacing[0] != None, "px cannot be None"
        assert spacing[1] != None, "py cannot be None"
        assert spacing[2] != None, "pz cannot be None"
        for i in range(slices.shape[-1]):
            v=np.sum(slices)*spacing[2]*spacing[0]*spacing[1]
    volume = v
    return volume

def resize3D(scan,shape, is_mask=False):
    """
    This function takes as input the scan to be resized and the desired shape and returns 
    the 3D volume after resizing it to the required shape.

    Parameters:
        nscan: 3D numpy array to resize. 
        shape: (new_depth,new_height,new_width)
    returns:
        (int/float) nscan: 3D numpy array
    """

    
    assert len(shape)==3,"The new shape should have 3 values"
    assert len(shape)==3, "Image should have 3 dimensions, given "+str(len(scan.shape))+ " dimensions"
    depth_factor=shape[0]/scan.shape[0]
    width_factor=shape[2]/scan.shape[2]
    height_factor=shape[1]/scan.shape[1]

    if is_mask==False:
        scan = ndimage.zoom(scan, (depth_factor, height_factor, width_factor), order=1)
    elif is_mask==True:
        scan = ndimage.zoom(scan, (depth_factor, height_factor, width_factor), order=1, mode = 'nearest' )


    return scan

