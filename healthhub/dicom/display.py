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

figsize = cmap = volume = v = vol1 = vol2 = vol3 = 0

def multiSliceViewer(vol ,fig_size=(100,100), c_map='gray'):
    """ 
    multiSliceViewer is for viewing volumetric image slices in jupyter or
    ipython notebooks. 

    User can interactively change the slice plane selection for the image and 
    the slice plane being viewed. 

    Parameters:
        (int/float) (3D numpy array) Vol = 3D numpy array of scan
        (int) fig_size = default(8,8), to set the size of the figure
        (string) c_map = default('gray'), string for the matplotlib colormap. You can find 
        more matplotlib colormaps on the following link:
        https://matplotlib.org/users/colormaps.html

    returns:
        none

    """
    global figsize, cmap, volume, v
    figsize = fig_size
    cmap = c_map 
    volume = vol


    v = [np.min(volume), np.max(volume)]
    ipyw.interact(__multiSlice)



def __multiSlice():
    global volume, vol1, vol2, vol3
    
    vol1 = np.transpose(volume, [1,2,0])
    vol2 = np.rot90(np.transpose(volume, [2,0,1]), 3) #rotate 270 degrees
    vol3 = np.transpose(volume, [0,1,2])
    maxZ1 = vol1.shape[2] - 1
    maxZ2 = vol2.shape[2] - 1
    maxZ3 = vol3.shape[2] - 1
    ipyw.interact(__plot_slice, 
        z1=ipyw.IntSlider(min=0, max=maxZ1, step=1, continuous_update=False, 
        description='Axial:'), 
        z2=ipyw.IntSlider(min=0, max=maxZ2, step=1, continuous_update=False, 
        description='Coronal:'),
        z3=ipyw.IntSlider(min=0, max=maxZ3, step=1, continuous_update=False, 
        description='Sagittal:'))

def __plot_slice(z1, z2, z3):
    global figsize, cmap, vol1, vol2, vol3, v
    # Plot slice for the given plane and slice
    f,ax = plt.subplots(1,3, figsize=figsize)
    ax[0].imshow(vol1[:,:,z1], cmap=plt.get_cmap(cmap), 
        vmin=v[0], vmax=v[1])
    ax[1].imshow(vol2[:,:,z2], cmap=plt.get_cmap(cmap), 
        vmin=v[0], vmax=v[1])
    ax[2].imshow(vol3[:,:,z3], cmap=plt.get_cmap(cmap), 
        vmin=v[0], vmax=v[1])
    plt.show()

def plotSlice(image):
    '''
    Displays the 2D plot of an image.

    Parameters:
            image (int)(float): A 2d numpy array

    Returns:
            none
    '''        
    plt.imshow(image,'gray')


def display3D_mask(volume):
    '''This function takes a 3D array of binary mask and returns the 3D plot of the mask.

    Parameters: 
            image (int)(float): A 3d numpy array

    Returns:

    '''

    p=volume.transpose(2,1,0)
    p = p[:,:,::-1]
    threshold=0.5
    alpha=0.5
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    face_color = [0.8, 0.2, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def intensityHist(array):

    '''
    This function takes an n dimensional numpy array as input and displays the plot of intensity histogram.       
    Parameters:
            image (int)(float): A 2d or 3d numpy array

    Returns: 
    '''
    array = np.array(array)

    [counts,bins,bars]=plt.hist(array.flatten())     
