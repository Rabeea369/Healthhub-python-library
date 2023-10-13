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

def as_png(image, path):
    '''
    This function saves 2D images in png
    Parameters:
        (int/float) (2D numpy array) image  : image to be saved  
        (string) path : output path


    Returns:
        none
    '''
    if not (path[-4:]=='.png'):
        path = path + '.png'
    if len(image.shape)<3:
        image = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    cv2.imwrite(path,image)
            
def as_jpeg(image, path):
    '''
    This function saves 2D images in jpeg
    Parameters:
        (int/float) (2D numpy array) image  : image to be saved  
        (string) path : output path


    Returns:
        none
    '''
    if not (path[-5:]=='.jpeg'):
        path = path + '.jpeg'
    if len(image.shape)<3:
        image = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    cv2.imwrite(path,image)
            
def as_bmp(image, path):
    '''
    This function saves 2D images in bmp
    Parameters:
        (int/float) (2D numpy array) image  : image to be saved  
        (string) path : output path


    Returns:
        none
    '''
    if not (path[-4:]=='.bmp'):
        path = path + '.bmp'
    if len(image.shape)<3:
        image = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(image)
    image.save(path)
            
def as_pdf(image, path):
    '''
    This function saves 2D images in pdf
    Parameters:
        (int/float) (2D numpy array) image  : image to be saved  
        (string) path : output path


    Returns:
        none
    '''
    if len(image.shape)<3:
        image = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(image)
    image.save(path + ".pdf")