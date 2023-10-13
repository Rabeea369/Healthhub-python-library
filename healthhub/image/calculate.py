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


def max_diameter_2D(mask,p_x, p_y ,arrow = False):
    ''' 
    This function finds the maximum diameter of objects in binary mask 2D image. 
    If arrow is True, an arrow will also be drawn to show the direction of the max diameter. So the function returns 
    the maximum diamter distance and in case of arrow =True, also returns the original image with arrow


    Parameters:
        (int) (2D numpy array) mask  :  the binary mask of the object for which the maximum diameter needs to be calculated
        (int/float) p_x : The pixel spacing in the x direction
        (int/float) p_y : The pixel spacing in the y diection


    Returns:
        (float) dist : the distance of the maximum diameter
        (int)(2D numpy array) image: image with arrow drawn to show direction of maximum diameter (optional)

    '''

    res0 = cv2.cvtColor(mask.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    res0 = np.where(res0>0,255,0)
    mask = np.where(mask>0,1,0)
    mask, num_lab = ndimage.label(mask)


    dist = []
    for i in range(num_lab):
        points = np.argwhere(mask==i+1)
        points = points[points[:,0].argsort()]
        distances = distance.cdist(points,points)
        if len(distances)>1:
            [p1,p2] = np.squeeze(np.where(distances == np.max(distances)))
            if isinstance(p1, np.ndarray): 
                p1 = p1[-1]
            if isinstance(p2, np.ndarray): 
                p2 = p2[-1]
            [ty,tx] = points[p1]
            [cy,cx] = points[p2]
            d = (math.sqrt((((ty-cy)*p_y)**2)+(((tx-cx)*p_x)**2)))
            dist.append(d)
        else:
            dist.append('none')
        if arrow==True:

            image = cv2.arrowedLine(res0.astype(np.uint8), (cx,cy), (tx,ty),(0,0,255),2)
            return dist,image
        else:
            return dist


def max_diameter_3D(full_mask,spacing):
    '''
    This function finds the max diameter of 3D binary object. It takes the 3D mask of the object, containing 
    only the slices that contain the object/nodule so shape can be like [6,45,45] where '6' is number of slices
    It also takes spacing which is an array containing pixel spacing of x/y and z axis, shape should be like [0.7,1.5]

    Parameters:
        (int) (3D numpy array) full_mask  : binary mask of object 
        (float) (list) spacing : pixel spacing along x/y and z axis


    Returns:
        (float) dist : the distance of the maximum diameter of object

    '''

    # Ful mask means that it should contain all the slices of nodule, so shape can be like [6,45,45]
    # spacing is array contains spacing of x,y and z axis, shape should be like [0.7,1.5]


    all_points = np.where(full_mask>0)
    x_coor = all_points[1]
    y_coor = all_points[2]
    z_coor = all_points[0]

    pairs = []
    coordinates = []

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
    for pair in pairs:
        p_1 = pair[0]
        p_2 = pair[1]
        distances[i] = math.sqrt(((p_1[0] - p_2[0])*spacing[0])**2 + ((p_1[1] - p_2[1])*spacing[0])**2 + ((p_1[2] - p_2[2])*spacing[1])**2)
        i = i+1
    max_dis = float(distances.max())
    return max_dis


def volume_3D(slices,spacing):
    '''
    This function calculates the 3D volume of an object mask. The mask should have the format axial (along 0th axis) ,
    coronal (along 1st axis),sagittal (along 2nd axis)
    The pixel spacing is z,y,x
    Parameters:
        (int)(3D numpy array) slices  : binary mask of 3D object 
        (float) (list) spacing : pixel spacing along z,y,x axis.


    Returns:
        (float) volume : volume of 3D object

    '''

    slices = np.where(slices>0,1,0)
    v = 0
    if len(slices.shape) != 3:
        v = np.sum(slices)*spacing[0]*spacing[1]
    else:
        p = np.argwhere(slices>0)
        p = np.amin(p[:,0])
        for i in range(p,slices.shape[0],1):
            a = np.sum(slices[i,:,:])*spacing[1]*spacing[2]
            v = v + a
    #         print('slice no.{i}'.format)
            if a==0:
                print('Done')
                break;
    volume = v
    return volume