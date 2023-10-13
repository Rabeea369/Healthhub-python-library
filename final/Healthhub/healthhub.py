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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, feature

from scipy.spatial import distance
import skimage

import glob as glob
import SimpleITK as sitk
from tqdm import tqdm
import cv2
import pydicom
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, feature
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import color
from skimage import io
import cv2
import math
from matplotlib import image
from matplotlib import pyplot
from scipy import ndimage

import numpy as np
import ipywidgets as ipyw
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir




class healthhub:
    def __init__(self):
        self.dicom=dicom()
        self.image=Image_class()
        self.CV=CV()
        
        
class dicom:

    def __init__(self):
        self.read=read()
        self.modify=modify()
        self.display=display()
        #self.process=process()
        
        
class display:
    """
    A class to represent display.

    ...

    Attributes
    ----------
    name : volume
        3d volume array
    

    Methods
    -------
    multipleSlice(volume):
        Plot the multiple slices of a given volume
    
    plotSlice(image):
        Plots the 2D plot of given slice image  
    
    display3D(volume):
        Plots the 3D plot given numpy array
    
    intensityHist(array):
        it will plot intensity histogram of given array        
           
    
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the diplay object.
        """

        pass
        
    def multipleSlice(self,volume):
        '''
        allows the user to navigate through different slices by using a slider widget
        
        Parameters:
        volume (int)(float): A 3d numpy array

        Returns:
                Mutiple slices (int)(float): axial , coronal and sagittal slices 
        '''

        figsize = (100,100)
        cmap = 'gray'
        v = [np.min(volume), np.max(volume)] 

        def views():
            def plot_slice(z1, z2, z3):
                # Plot slice for the given plane and slice
                f,ax = plt.subplots(1,3, figsize=figsize)
                #print(self.figsize)
                #self.fig = plt.figure(figsize=self.figsize)
                #f(figsize = self.figsize)
                ax[0].imshow(vol1[:,:,z1], cmap=plt.get_cmap(cmap), 
                    vmin=v[0], vmax=v[1])
                ax[1].imshow(vol2[:,:,z2], cmap=plt.get_cmap(cmap), 
                    vmin=v[0], vmax=v[1])
                ax[2].imshow(vol3[:,:,z3], cmap=plt.get_cmap(cmap), 
                    vmin=v[0], vmax=v[1])
                plt.show()

            vol1 = np.transpose(volume, [1,2,0])
            vol2 = np.rot90(np.transpose(volume, [2,0,1]), 3) #rotate 270 degrees
            vol3 = np.transpose(volume, [0,1,2])
            maxZ1 = vol1.shape[2] - 1
            maxZ2 = vol2.shape[2] - 1
            maxZ3 = vol3.shape[2] - 1
            ipyw.interact(plot_slice, 
                z1=ipyw.IntSlider(min=0, max=maxZ1, step=1, continuous_update=False, 
                description='Axial:'), 
                z2=ipyw.IntSlider(min=0, max=maxZ2, step=1, continuous_update=False, 
                description='Coronal:'),
                z3=ipyw.IntSlider(min=0, max=maxZ3, step=1, continuous_update=False, 
            description='Sagittal:'))

        ipyw.interact(views)

    def plotSlice(self, image):
        '''
        Returns the 2D plot of given slice image        
        Parameters:
                image (int)(float): A 2d numpy array

        Returns:
                Plot an Image  
        '''        
        plt.imshow(image,'gray')
        

    def display3D(self,volume):
        '''Returns the 3D plot given numpy array
        
        Parameters:
                image (int)(float): A 3d numpy array

        Returns:
                Plot 3d Image  
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

    def intensityHist(self,array):
        
        '''
        it will plot intensity histogram of given array        
        Parameters:
                image (int)(float): A 2d or 3d numpy array

        Returns:
                Plots Image histogram  
        '''
        array = np.array(array)

        [counts,bins,bars]=plt.hist(array.flatten())     
        
class read:
    """
    A class to represent read.

    ...

    Attributes
    ----------
    name : None
    

    Methods
    -------
     
    scan(folder_path): 
        it will  take path of scan as input and return scan array in 4d and its metadeta
        
    Slice(slice_path):
        it will  take path of slice as input and return slice array  and its metadeta
        
    tag(tags,ds):
         it will  return the given tags values 
        
    getSeriesPlane(ds):
         it will  return the  tags with in getSeriesPlane 
    
    orientation(self,orientation_tag_value): 
         it will  return orientation tag value as a type string
         
    seriesData(ds):   
         it will  returns a list of sopseriesUID, series_number, view corresponding to each series

    
    getImagePlane(self,meta_data):  
         it will  returns  all the tags within ImagePlane  module
         
    getFrameOfReference(self,meta_data): ##Public func
         it will  returns  all the tags within FrameOfReference  module
         
    getPatientStudy(self,meta_data):
         it will  returns  all the tags within getPatientStudy  module
         
    getGeneralStudy(self,meta_data): 
         it will  returns  all the tags within getGeneralStudy  module
        
    getImagePixel(meta_data):    
          it will  returns  all the tags within getImagePixel  module
        
        
    """

    def __init__(self):#, folder_path):
        """
        Constructs all the necessary attributes for the diplay object.

        Parameters
        ----------
            name : None
            
        """

        pass
        #super().__init__()

    def scan(self,folder_path): ##Public func
        '''
        it will  take path of scan as input and return scan array in 4d and its metadeta
        
        Parameters:
                folder_path : path of given dicom scan

        Returns:
                (int)(float): Scan 
                            : Meta Data
        '''        
       
        
        files = self.sorted_files(folder_path)
        f, myscan, mytags = self.read_scan(folder_path,files)
        return myscan, mytags

    def sorted_files(self,folder): ##Private function
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


    def read_scan(self,folder_path,files): ##Private func
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
            myscan.append(img)
            f.append(file)
        return f, myscan, mytags

    def Slice(self, slice_path): ##Public
        '''
        it will  take path of slice as input and return slice array  and its metadeta
        
        Parameters:
                folder_path : path of given dicom scan

        Returns:
                (int)(float): Slice 
                            : Meta Data
        '''        
        ds = pydicom.dcmread(slice_path)
        numpyImage = ds.pixel_array
        return numpyImage, ds

    #def seriesPlaneData(self): ##public
    #    '''it will  display module series plane data'''
    #    print('Series Orientation: ',self.orient)
    #    print('Series Instance UID: ',self.series_ins_uid)
    #    return self.orient,self.series_ins_uid

    def tag(self,tags,ds): ##Public
        '''it will  return the given tags values 
        
        Parameters:
                tags : any tags of given dicom scan
                ds : meta deta of dicom scan

        Returns:
                (int)(float)(String): Tag Value 
                            
        '''
        (x1,x2) = tags
        if (x1,x2) in ds:
            print(ds[x1,x2])
            return ds[x1,x2].value
        else:
            print('Tag not found')
    
    ###
    def getSeriesPlane(self,ds): ##public
        '''it will  return the  tags with in getSeriesPlane 
        
        Parameters:
                ds= meta deta
                
        Returns:
                Series Orientation, Series Instance UID                            
        '''
        l=[]
        if ((0x0020,0x0037) in ds):
            o=ds[0x0020,0x0037].value
            
        else:
            o=''
        if ((0x0020,0x000e) in ds):
            s=ds[0x0020,0x000e].value
            #print("hi")
        else:
            s=''
        l.append([o,s])

        return l

    

    def orientation(self,orientation_tag_value): ##Public func
        '''
        it will  return orientation tag value as a type string
        
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

    def seriesData(self,ds):   ##Public func
        '''it will  returns a list of sopseriesUID, series_number, view corresponding to each series
         Parameters:
                ds  :  scan meta deta

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
            #b = ds[i][0][0x0020,0x0037].value if ((0x0020,0x0037) in ds[i][0]) else ''
            c = ds[i][0][0x0020,0x0011].value if ((0x0020,0x0011) in ds[i][0]) else ''
            l.append([a,b,c])

        return l




    def getImagePlane(self,meta_data):  ##Public func
        #Pixel Spacing
        '''it will  returns  all the tags within ImagePlane  module
         Parameters:
                ds  :  scan meta deta

        Returns:
                table : Tag Name, Tag Code and Tag Values with in imagePlane Module 
                        
        '''
        data =meta_data.group_dataset(0x0020)


        #col_names = ["Tag Name", "Tag Code","Tag Value"]
        #print(tabulate(data, headers=col_names))
        return data



    def getFrameOfReference(self,meta_data): ##Public func
        '''it will  returns  all the tags within FrameOfReference  module
         Parameters:
                ds  :  scan meta deta

         Returns:
                table : Tag Name, Tag Code and Tag Values with in FrameOfReference Module 
                        
         '''
        data = meta_data.group_dataset(0x0020)

       # print(data)
        return data

    def getPatientStudy(self,meta_data):
        '''it will  returns  all the tags within getPatientStudy  module
         Parameters:
                ds  :  scan meta deta

         Returns:
                table : Tag Name, Tag Code and Tag Values with in getPatientStudy Module 
                        
         '''   
        data=meta_data.group_dataset(0x0010)

        #col_names = ["Tag Name", "Tag Code","Tag Value"]

        #print(data)
        return data


    def getGeneralStudy(self,meta_data): ##Public func
        '''it will  returns  all the tags within getGeneralStudy  module
         Parameters:
                ds  :  scan meta deta

         Returns:
                table : Tag Name, Tag Code and Tag Values with in getGeneralStudy Module 
                        
        '''   
       

        data =meta_data.group_dataset(0x0008) 

        #print(data)
        return data

    def getImagePixel(self,meta_data):    ##Public func
        '''it will  returns  all the tags within getImagePixel  module
         Parameters:
                ds  :  scan meta deta

         Returns:
                table : Tag Name, Tag Code and Tag Values with in getImagePixel Module 
                        
        '''
        data = meta_data.group_dataset(0x0028) 

            #    ["Bits Stored ", "(0008,0101)",meta_data[0x0028,0x0101].value],
             #   ["High Bit ", "(0008,0102)",meta_data[0x0028,0x0102].value],
             #   ["Pixel Representation ", "(0008,0103)",meta_data[0x0028,0x0103].value]]
        #col_names = ["Tag Name", "Tag Code","Tag Value"]

        #print(data)
        return data
    
    
class modify:
    """
    A class to represent modify.

    ...

    Attributes
    ----------
    name : None
    

    Methods
    -------
    
    tag(tags,ds,value):
        it will  modify   tags  values in scan meta data
         
    image(ds,image):
        This function saves the provided image (rgb or gray scale) in the dataset
       


    maxDiameter(full_mask,spacing):
        This function finds the max diameter of 3D object
         
    """

    def __init__(self):
        
        pass
        """
        Constructs all the necessary attributes for the diplay object.

        Parameters
        ----------
            name : None
            
        """


    def tag(self,tags,ds,value):
        '''it will  modify   tags  values in scan meta data
         Parameters:
                ds  :  scan meta deta
                tags : tag number to modify
                value : new value for the tag

        Returns:
                ds : Meta data with new tag value 
                        
        '''        
        (x1,x2) = tags
        if (x1,x2) in ds:
            print('tag name: ',ds[x1,x2].keyword)
            print('previous value: ',ds[x1,x2].value)
            ds[x1,x2].value = value
            print('new value: ',ds[x1,x2].value)
        else:
            print('Tag not found')
        return ds

    def image(self,image,output_path,dummy_dcm_path=''):
        '''
        This function saves the provided image (rgb or gray scale) in the dataset
        Parameters:
                ds  :  scan meta deta
                image : numpy array
                

        Returns:
                ds : Meta data with image value 
                        
        '''
        if dummy_dcm_path=='':
            dummy_dcm_path = 'dummy.dcm'
        
        ds = pydicom.dcmread(dummy_dcm_path)
        
        ds.Rows = image.shape[0]
        ds.Columns = image.shape[1]

        if len(image.shape)>2: ##rgb
            image = image.astype(np.uint8)
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
        else:
            image = image.astype(np.int16)
        if (0x7fe0, 0x0010) in ds:
            ds.PixelData = image.tobytes()
        else:
            ds.add_new([0x7fe0, 0x0010], 'OB', image.tobytes())
        ds.save_as(output_path)



    def maxDiameter(full_mask,spacing):
        
        '''
        This function finds the max diameter of 3D object
         Parameters:
                full_mask  : it should contain all the slices of nodule
                spacing : pixel spacing
                

         Returns:
                ds : Meta data with image value 
                        
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
            distances[i] = math.sqrt(((p_1[0] - p_2[0])*spacing[0])**2 + ((p_1[1] - p_2[1])*spacing[0])**2 + ((p_1[2] - p_2[2])*spacing[1])**2)
            i = i+1
        max_dis = float(distances.max())
        max_p = np.where(distances==distances.max())[0]
        max_pints = []
        for p in max_p:
            max_pints.append(pairs[p])
        return max_dis, max_pints
    
    
    
class process:
    """
    A class to represent process.

    ...

    Attributes
    ----------
    name : None
    

    Methods
    -------
    
    volumeCalculator(slices,spacing):
        Returns the volume of an object in 3D binary mask in physical measurements.
    
    getNormalized(scan,wc ,wl):
        Returns the normalized volume.

    windowing(self, scan,window_mode, window):
        apply window function on the scan
    """



    def __init__(self):
        """
        Constructs all the necessary attributes for the diplay object.

        Parameters
        ----------
            name : None
            
        """

        self.segmentation =segmentation()   
    
    def volumeCalculator(self,slices,spacing):
        """
        Returns the volume of an object in 3D binary mask in physical measurements.

        This function takes 3D binary mask, of an object as an input along with pixel spacing and returns the physical volume as an output.     The unit of volume returned is same as the unit of pixel spacing (which is usually mm).

        :param slices: Slices containing binary mask, usually of tumor. 
        :type slices: int
        :param spacing: Takes in a list of pixel spacing as [px,py,pz] where px,py and pz are pixel spacings in x,y and z cordinates. 
        :type spacing: list

        :return: Returns the physical volume
        :rtype: float
        """
        #def __init__(self, spacing=[None,None,None]):
        self.spacing[0]=spacing[0]
        self.spacing[1]=spacing[1]
        self.spacing[2]=spacing[2]


        v=0

        if len(slices.shape) == 2:
            assert len(self.spacing) != 2, "Spacing should have two values, spacing should be a list of two numbers in case of 2D image"

            assert self.spacing[0] != None, "px cannot be None"
            assert self.spacing[1] != None, "py cannot be None"

            v = np.sum(slices)*self.spacing[0]*self.spacing[1]
        else:
            assert len(self.spacing) == 3, "pz should be defined"

            assert self.spacing[0] != None, "px cannot be None"
            assert self.spacing[1] != None, "py cannot be None"
            assert self.spacing[2] != None, "pz cannot be None"
            for i in range(slices.shape[-1]):
                v=np.sum(slices)*self.spacing[2]*self.spacing[0]*self.spacing[1]

        return v
    
    def windowing(self, scan,window_mode, window):
        """
        apply window function on the scan
        
        :param scan: input scan to be performed windowing on.
        :type number: pydicom.dataset.FileDataset
    
        :return: The result of windowing.
        :rtype: numpy.ndarray
        """
        assert window_mode=='min/max' or window_mode=='wl/ww' or window_mode=='scan_min/scan_max', "window_mode should be defined!"
        assert len(window)==2, "Please define min/max or ww/wl values for windowing!"
    
        import numpy as np
        if window_mode=='wl/ww':
            assert window[0]!=None and self.window[1]!=None, "WW/WL values cannot be None!" 
            wl=self.window[0]
            ww=self.window[1]
            mn = wl - ww/2
            mx = wl + ww/2
        elif self.window_mode=='min/max':
            assert self.window[0]!=None and self.window[1]!=None, "Min/max values cannot be None!"
            mn=self.window[0]
            mx=self.window[1]
        elif self.window_mode=='scan_min/scan_max':
            mn = np.amin(scan)
            mx = np.amax(scan)
        
        d = mx - mn
        scan  = np.where(scan<mn,mn, scan)
        scan  = np.where(scan>mx,mx, scan)
        scan = (scan-mn)/d
        return scan
        


    def getNormalized(self,scan,wc ,wl):
        """
        Returns the normalized volume.

        :param scan: 3D image to normalize. 
        :type nscan: int/float
        """
        mn = wc - wl
        mx = wc+wl
        d = mx - mn
        scan  = np.where(scan<mn,mn, scan)
        scan  = np.where(scan>mx,mx, scan)
        scan = (scan-mn)/d
        return scan 

    def resize3D(self, nscan,shape):
        """
        Returns the 3D volume after resizing it to the required shape.

        :param nscan: 3D image to resize. 
        :type nscan: int/float
        """
        
        from scipy import ndimage
        assert len(shape)==3,"The new shape should have 3 values"
        assert len(shape)==3, "Image should have 3 dimensions, given "+str(len(nscan.shape))+ " dimensions"
        depth_factor=shape[0]/nscan.shape[0]
        width_factor=shape[2]/nscan.shape[2]
        height_factor=shape[1]/nscan.shape[1]

        if mask==False:
            nscan = ndimage.zoom(nscan, (depth_factor, height_factor, width_factor), order=1)
        elif is_mask==True:
            nscan = ndimage.zoom(nscan, (depth_factor, height_factor, width_factor), order=1, mode = 'nearest' )
    
    
        return nscan
        

class segmentation:
    """"

    Attributes
    ----------
    name : None
    

    Methods
    -------
    
    generic_window(ds):
        
    boneSeg(scan,wc ,wl):
      
    seg():
        """
   
    
    def __init__(self, path):
        self.path=path
        
    def genericWindow(self,ds):
        
        def get_normalized(scan,wc ,wl):
            mn = wc - wl
            mx = wc+wl
            d = mx - mn
            scan  = np.where(scan<mn,mn, scan)
            scan  = np.where(scan>mx,mx, scan)
            scan = (scan-mn)/d
            return scan  
        
        wc= ds[0x28,0x1050].value  #window center
        w=(ds[0x28,0x1051].value) #window width
        ww=int(w/2)
       
        new_wc=wc+(0.3*wc)
        new_ww=(0.8*ww)   
        
        img=ds.pixel_array
        result=get_normalized(img,wc+(0.3*wc),(0.8*ww) )
     
        return result
    
    def boneSeg(self,ds):
    
        def normal(un):
            mn = np.amin(un)
            mx = np.amax(un)
            d = mx - mn
            normalized = (un-mn)/d
            return normalized
        
        def get_normalized(scan):
            wc= (300-(-1000))/(1000-(-1000)) #normalizing WC
            w= (1000-(-1000))/(1000-(-1000))  #normalizing WW
            wl=w/2
            mn = wc - wl
            mx = wc+wl
            d = mx - mn
            scan  = np.where(scan<mn,mn, scan)
            scan  = np.where(scan>mx,mx, scan)
            scan = (scan-mn)/d
            return scan        

        img=ds.pixel_array
        normal_img=normal(img)
        final_image=get_normalized(normal_img)

        return final_image     
    
    def seg(self):
        
        import pydicom
        ds=pydicom.dcmread(self.path)
        #img=ds.pixel_array
        try: 
            wc= ds[0x28,0x1050].value  #window center
            w=ds[0x28,0x1051].value #window width
            res = self.generic_window(ds)
        except:
            res = self.bone_seg(ds)
        
        return res
    
    
class Image_class:
    """
    A class to represent Image.

    ...

    Attributes
    ----------
    name : None
    

    Methods
    -------
    info(additional=""):
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the diplay object.

        Parameters
        ----------
            name : None
            
        """

        
        self.calculate = self.calculate_class()
        self.save = self.save_class()
        self.MIP = self.MIP_class()
    
    class calculate_class:
        def __init__(self):
            pass
        def max_diameter_2D(self,mask,p_x, p_y ,arrow = False):
            ''' 
            This function finds the maximum diameter of objects in binary mask image. 
            If arrow is True, an arrow will be drawn to show the direction of the max diameter. 
            
            
            Parameters:
                mask  :  
                p_x : 
                p_y :
                

            Returns:
                dist : 
                        
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
                    
                    res0 = cv2.arrowedLine(res0.astype(np.uint8), (cx,cy), (tx,ty),(0,0,255),2)
                    return dist,res0
                else:
                    return dist
        
        
        def max_diameter_3D(self,full_mask,spacing):
            '''
            This function finds the max diameter of 3D object
            Ful mask means that it should contain all the slices of nodule, so shape can be like [6,45,45] where '6' is number of slices
            spacing is array contains spacing of x,y and z axis, shape should be like [0.7,1.5]
            
            Parameters:
                full_mask  :  
                spacing : pixel spacing
                

            Returns:
                dist : 
                        
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
        
        
        def volume_3D(self,slices,spacing):
            '''
            This function calculates the 3D volume of an object mask. The mask should have the format axial,coronal,sagittal
            The pixel spacing is z,y,x
            Parameters:
                slices  :  
                spacing : pixel spacing
                

            Returns:
                dist : 
                        
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
            return v
        
        def maxNodule(self,mask_scan):
            '''
            Depending on the intensities,
            this function will only return the nodule which is enhanced and more visible in the scans and
            remove the other nodules function saves 2D images with accordance with types with the types
            
            Parameters:
                    scan :  
                    mask :


                Returns:
                    processed mask : 
            '''

            labelled_mask, num_labels = skimage.measure.label(mask_scan)
            mask_scan = np.zeros(mask_scan.shape)
            for j in range(1,num_labels+1):
                mask2 =labelled_mask_scan.copy()
                mask2[mask2!=j]=0
                mask2[mask2==j]=1
                if(np.sum(mask2)>np.sum(mask_scan)):
                    mask_scan = mask2
            return(mask_scan)

    class save_class:
        '''
        This function saves 2D images with accordance with types with the types
        Parameters:
                image  :  
                path :
                

            Returns:
                image : 
                        
            
        
        '''
        def __init__(self):
            pass
        def as_png(self,image, path):
            if len(image.shape)<3:
                image = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2RGB)
            cv2.imwrite((path + ".png"),image)
            
        def as_jpeg(self,image, path):
            if len(image.shape)<3:
                image = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2RGB)
            cv2.imwrite((path + ".jpeg"),image)
            
        def as_bmp(self,image, path):
            if len(image.shape)<3:
                image = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image)
            image.save(path + ".bmp")
            
        def as_pdf(self,image, path):
            if len(image.shape)<3:
                image = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image)
            image.save(path + ".pdf")
    
    class MIP_class:
        '''
        This class generates Maximum Intensity Projections along the given directions
        Parameters:
                image  :  
                path :
                

            Returns:
                image : 
                        
                       
        
        '''
        
        def axial(self,scan):
            image = np.amax(scan, axis = 0)
            return(image)
        def sagittal_left(self,scan):
            scan = np.array(scan)
            image = np.amax(scan[:,:,round(scan.shape[2]/2):], axis = 2)
            return(image)
        def sagittal_right(self,scan):
            scan = np.array(scan)
            image = np.amax(scan[:,:,:round(scan.shape[2]/2)], axis = 2)
            return(image)
        def sagittal(self,scan):
            scan = np.array(scan)
            image = np.amax(scan, axis = 2)
            return(image)
        def coronal(slf, scan):
            scan = np.array(scan)
            image = np.amax(scan, axis = 1)
            return(image)
    
    def mask_overlay(self, img, mask, mode = 'axial'):
        '''
        This function overlays the mask on the scan with a MIP in the provided direction: axial, coronal or sagittal
        Parameters:
                image  :  
                mask :
                mode :
                

            Returns:
                image : 
                        
                       
        
        '''
       
        
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
        
        
            
    
class CV:
    """
    A class to represent Computervision.

    ...

    Attributes
    ----------
    name : None
    

    Methods
    -------
    EdgeDetection(img_path):
        Takes in an Image path,displays the edges of the image 
    
    ImageSegmentation(img_path): 
        Image segmentation using K-Means takes input as image path and displays segmentation of image
    
    TemplateMatching(img_path,temp_path):
        It simply slides the template image over the input
        image (as in 2D convolution) and compares the template and patch of input image under the template image.
     
     BlobDetector(img_path):
        Takes in an image path and Draw detected blobs as blue circles
    
    OpticalFlow(video_path):
        Optical flow is the motion of objects between the consecutive frames of the sequence, 
        caused by the relative motion between the camera and the object.
        It takes motion video as input in avi format and detects Moving objects
        
        

    """

    def __init__(self):
                
        """
        Constructs all the necessary attributes for the diplay object.

        Parameters
        ----------
            name : None
            
        """
        pass
    

    def EdgeDetection(self, img_path):
        '''Takes in an Image path,displays the edges of the image
         Parameters:
                img_path  :  path of the image
               

        Returns:
                edges : Plot the edges in image 
                        
        '''
        img = cv2.imread(img_path)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(gray_img.shape)

        # applying canny edge transformations
        edges = cv2.Canny(gray_img, threshold1=30, threshold2=100)

            # showing the output frame
        plt.imshow(edges,'gray')
        
    def ImageSegmentation(self,img_path): 
        '''Image segmentation using K-Means takes input as image path and displays segmentation of image
         Parameters:
                img_path  :  path of the image
               

        Returns:
                image segmentation : Plot the segmented image
                        
        '''
        img = cv2.imread(img_path)
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
        plt.imshow(segmented_image)
        
    def TemplateMatching(self,img_path,temp_path):
        ''''
        It simply slides the template image over the input
        image (as in 2D convolution) and compares the template and patch of input image under the template image.

         Parameters:
                img_path  :  path of the image
                temp_path : path of template image
               

        Returns:
                image  : Plot the template matching image
                        
        '''
        
        img = cv2.imread(img_path)
        temp = cv2.imread(temp_path, 0)


        source_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        w,h = temp.shape[::-1]

        res = cv2.matchTemplate(source_gray, temp, cv2.TM_CCOEFF_NORMED)
        threshold = 0.2
        loc = np.where(res >= threshold)

        print(*loc)
        for port in zip(*loc[::-1]):
            a=cv2.rectangle(img, port,(port[0] + w, port[1] + h), (0, 255, 255), 2)


        plt.imshow(a)
        
    def BlobDetector(self,img_path):
        ''' Takes in an image path and Draw detected blobs as blue circles
        Parameters:
                img_path  :  path of the image
               

        Returns:
                Blobs : Plot the blobs detected in image
                        
        '''
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Set up the blob detector
        detector = cv2.SimpleBlobDetector_create()

        # Detect blobs from the image.
        keypoints = detector.detect(img)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS - This method draws detected blobs as red circles and ensures that the size of the circle corresponds to the size of the blob.
        blobs = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        plt.imshow(blobs)
        
    def OpticalFlow(self,video_path):
        '''
        Optical flow is the motion of objects between the consecutive frames of the sequence, 
        caused by the relative motion between the camera and the object.
        It takes motion video as input in avi format and detects Moving objects
        Parameters:
                video_path  :  path of the motion video
               

        Returns:
                images : Plot the points on image with relative motion between the consecutive frames of the sequence
                        
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

        




        
        