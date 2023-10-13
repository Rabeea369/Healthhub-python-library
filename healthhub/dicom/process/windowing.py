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

def __get_windowed(scan, wc, wl):
    mn = wc - wl
    mx = wc + wl
    d = mx - mn
    scan  = np.where(scan<mn,mn, scan)
    scan  = np.where(scan>mx,mx, scan)
    scan = (scan-mn)/d
    return scan  

def manual(scan,window_mode, window):
    """
    This function applies windowing/grey-level mapping on the scan. It takes as input the 3D scan, 
    the window mode ('min/max' or 'wl/ww' or 'scan_min/scan_max')  , and the window to be applied ([min, max] or [wl, ww]).

    Parameters: 
        (int/float) (3D numpy array) scan: a 3D numpy array on which windowing needs to be performed.
        (string) window mode : a string defining the type of windowing
        (2x1 numpy array) window : a numpy array containing the minimum/window level on 0th index 
                                    and maximum/window centre of the window to be applied on 1st index.
    returns:
        (int/float) (3D numy array) scan: The result of windowing on the original scan.
    """
    assert window_mode=='min/max' or window_mode=='wl/ww' or window_mode=='scan_min/scan_max', "window_mode should be defined!"
    assert len(window)==2, "Please define min/max or ww/wl values for windowing!"

    import numpy as np
    if window_mode=='wl/ww':
        assert window[0]!=None and window[1]!=None, "WW/WL values cannot be None!" 
        wl= window[0]
        ww= window[1]
        mn = wl - ww/2
        mx = wl + ww/2
    elif window_mode=='min/max':
        assert window[0]!=None and window[1]!=None, "Min/max values cannot be None!"
        mn=window[0]
        mx=window[1]
    elif window_mode=='scan_min/scan_max':
        mn = np.amin(scan)
        mx = np.amax(scan)

    d = mx - mn
    scan  = np.where(scan<mn,mn, scan)
    scan  = np.where(scan>mx,mx, scan)

    return scan



def generic_window(scan, ds):
    '''
    This function takes in the scan and the metadata of a single slice and applies the window specified in that metadata on the scan.

    Parameters:
    (int/float) (3D numpy ndarray) scan: The input scan on which windowing needs to be applied
    (File Dataset object) ds: This is the metadata of the file on which windowing needs to be applied

    returns:
    (int/float) (2D numpy array) result: The image after windowing
    '''
    wc= int(ds[0x28,0x1050].value)  #window center
    w=int(ds[0x28,0x1051].value) #window width
    wl=int(w/2)
    result=__get_windowed(scan,wc,wl)

    return result





