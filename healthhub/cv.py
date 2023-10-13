import numpy as np
import glob as glob
import SimpleITK as sitk
from tqdm import tqdm
import cv2
import pydicom
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
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

def EdgeDetection(img_path):
    '''This function takes in an Image path and returns the edges of the image in a new image.
     Parameters:
            (string) img_path  :  path of the image


    Returns:
            (int/float)(2D numpy array) edges : image containing edges 

    '''
    img = cv2.imread(img_path)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray_img.shape)

    # applying canny edge transformations
    edges = cv2.Canny(gray_img, threshold1=30, threshold2=100)
    return(edges)

def ImageSegmentation(img_path): 
    '''This function performs Image segmentation using K-Means and takes input the image path and returns the segmentation of image
     Parameters:
            (string) img_path  :  path of the image


    Returns:
            (int/float)(2D numpy array) segmented_image : image segmented into different clusters

    '''
    
    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        # reshape the image to a 2D array of pixels and 3 array values
    pixels_values = img.reshape((-1, 3))

    # conerting to float32
    pixels_values = np.float32(pixels_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Number of clusters]
    K = 3

    _, labels, (centers) = cv2.kmeans(pixels_values,
                                      K=K,
                                      bestLabels=None,
                                      criteria=criteria,
                                      attempts=10,
                                      flags=cv2.KMEANS_RANDOM_CENTERS)

    # converting to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels]

    segmented_image = segmented_image.reshape(img.shape)
    return(segmented_image)

# def TemplateMatching(img_path,temp_path):
#     ''''
#     It simply slides the template image over the input
#     image (as in 2D convolution) and compares the template and patch of input image under the template image.

#      Parameters:
#             img_path  :  path of the image
#             temp_path : path of template image


#     Returns:
#             image  : Plot the template matching image

#     '''

#     img = cv2.imread(img_path)
#     temp = cv2.imread(temp_path, 0)
    
#     plt.imshow(temp)
#     print(temp.shape)
#     source_gray = img
#     if len(img.shape)>2:
#         source_gray = cv2.cvtColor(source_gray, cv2.COLOR_BGR2GRAY)
#     if len(temp.shape)>2:
#         temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

#     w,h = temp.shape[::-1]

#     res = cv2.matchTemplate(source_gray, temp, cv2.TM_CCOEFF_NORMED)
#     threshold = 0.2
#     loc = np.where(res >= threshold)

#     print(*loc)
#     for port in zip(*loc[::-1]):
#         a=cv2.rectangle(img, port,(port[0] + w, port[1] + h), (0, 255, 255), 2)

#     return(a)

def BlobDetector(img_path):
    ''' This function takes in an image path and Draws detected blobs as blue circles and returns this new image
    Parameters:
            (string) img_path  :  path of the image


    Returns:
            (int/float)(2D numpy array) blobs : the blobs detected in image

    '''
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Set up the blob detector
    detector = cv2.SimpleBlobDetector_create()

    # Detect blobs from the image.
    keypoints = detector.detect(img)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS - This method draws detected blobs as red circles and ensures that the size of the circle corresponds to the size of the blob.
    blobs = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    return(blobs)

def OpticalFlow(video_path):
    '''
    Optical flow is the motion of objects between the consecutive frames of the sequence, 
    caused by the relative motion between the camera and the object.
    It takes motion video path as input in avi format and displays images with Moving objects detected.
    Parameters:
            (string) video_path  :  path of the motion video


    Returns:
            
    '''

    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    gs_im0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points_prev = cv2.goodFeaturesToTrack(gs_im0, 100, 0.03, 9.0, False)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
        #is_empty = frame.size == 0

        #print(is_empty)

            gs_im1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            # Call tracker.
            points, st, err = cv2.calcOpticalFlowPyrLK(gs_im0, gs_im1, points_prev, None, (3,3))
            #print(points)
            for i,p in enumerate(points):
                a,b = p.ravel()
                frame = cv2.circle(frame,(int(a),int(b)),3,(255,255,255),-1)

            plt.imshow(frame)
            plt.show()
            points_prev = points
            gs_im0 = gs_im1


        else:
            break
           # print(frame)





