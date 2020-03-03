import pydicom
import numpy as np
import pandas as pd 
import os
from skimage import measure, morphology
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
import tensorflow as tf
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from multipledispatch import dispatch
import sys
sys.path.append(".")
import time
from keras.preprocessing.image import ImageDataGenerator
from mammo2risk.dicom_manager import DicomManager

import warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn

class Preprocessor(object) : 
    BACKGROUND_REGION = -1 # Air region 
    BREAST_REGION = 0 # Non-dense breast region
    DENSE_REGION = 1 # Conventional dense region 
    AC_DENSE_REGION = 2 # Higher dense region
    CC_DENSE_REGION = 3 # Highest dense region 
    HOLOGIC_THRESHOLD = 50 # Predefined threshold for Hologic's mammograms
    GE_SCALE_FACTOR = 0.0001*1914*2294
    HOLOGIC_SCALE_FACTOR = 0.000049*3328*2560
    
    def __init__(self, width, height, interpolation) : 
        self.width = width
        self.height = height
        self.interpolation = interpolation
        self._param = {'width':width, 'height':height, 'interpolation':interpolation} 
    
    def print_info(self): 
        print(type(self).__name__)
        print(self._param)
        
    @dispatch(np.ndarray)  
    def resize_image(self, image) : 
        """ resize image with 2D interpolation 

        Args:
            image (ndarray)
            width (int): width
            height (int): height
            interpolation (int): interpolation method 
              range 0-5 with the following semantics: 
            - 0: Nearest-neighbor 
            - 1: Bi-linear (default) 
            - 2: Bi-quadratic 
            - 3: Bi-cubic 
            - 4: Bi-quartic 
            - 5: Bi-quintic
        Returns:
            ndarray: resized image
        """
        image = image.reshape([image.shape[0], image.shape[1]])
        resized_image = resize(image, (self.width, self.height), preserve_range=True, order=self.interpolation)
        return resized_image
      
    @dispatch(str)  
    def resize_image(self, file) : 
        dicom = self.load_dicom(file)
        image = self.get_dicom_pixel(dicom)
        return self.resize_image(image)
      
    @classmethod
    def get_image(cls, file, width=256, height=224, interpolation=3) : 
        """ get image with from dicom file with given size 

        Args:
            file (string): file path
            width (int): width
            height (int): height
        Returns:
            ndarray: resized image
        """
        dicom = cls.load_dicom(file)
        image = cls.get_dicom_pixel(dicom)
        return Preprocessor(width,height,interpolation).resize_image(image)
    
    @classmethod
    def load_dicom(cls, path) : 
        return pydicom.read_file(path, force=True)   
    
    @classmethod
    def get_dicom_pixel(cls, dicom) : 
        """ get full size with from dicom file
        
        Args:
            dicom (dicom object)
        Returns:
            ndarray: image
        """
        try : 
            result = dicom.pixel_array
        except : 
            dicom.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            result = dicom.pixel_array
            raise Exception(dicom.PatientID+ "has problem. ")
        finally : 
            return result
          
    @classmethod
    def get_pda(cls, image) :
        """ get percent density from segmented image
        
        Args:
            image (ndarray) : segmented image
        Returns:
            float: percent density
        """
        not_dense_pixels_sum = np.sum(image == cls.BREAST_REGION)
        dense_pixels_sum = np.sum(image == cls.DENSE_REGION)
        density = dense_pixels_sum/(dense_pixels_sum+not_dense_pixels_sum)
        
        return density    

    @classmethod
    def get_da(cls, image) :
        """ get dense area from segmented image
        
        Args:
            image (ndarray) : segmented image (2-D)
        Returns:
            float: dense are
        """
        dense_pixels_sum = np.sum(image == cls.DENSE_REGION)
        result = dense_pixels_sum*cls.GE_SCALE_FACTOR/(image.shape[0]*image.shape[1])
        
        return result
    
    @classmethod 
    def get_ba(cls, file, width, height):
        image = cls.get_image(file, width, height)
        bg_threshold = cls.get_bg_threhold_by_manufacturer(file) 
        masked_image = cls.mask_air_region(image.copy(), image.copy(), background_threshold=bg_threshold)  
        air_region = cls.get_air_region(image=masked_image, threshold=bg_threshold) 

        ba = np.sum(air_region == 0)*cls.GE_SCALE_FACTOR/(width*height)
        return ba
      
    @classmethod
    def get_bas(cls, files, width, height) :
        """ get breast area from file
        
        Args:
            files (list) : dicom files
        Returns:
            float: bresat area
        """
        return list(map(lambda file: cls.get_ba(file, width, height), tqdm(files)))
      
    @classmethod
    def get_multilevel_pda(cls, image) :
        """ get percent densities (Cumulus, Altocumulus, Cirrocumulus) from segmented image
    
        Args:
            image (ndarray, [w,h,4]) : segmented image with label 0,1,2,3 
             0 : nondense breast region 
             1 : dense region
             2 : higher dense region
             3 : highest dense region
        Returns:
            float: percent density (%)
        """
        not_dense_pixels_sum = np.sum(image == cls.BREAST_REGION)
        dense_pixels_sum = np.sum(image == cls.DENSE_REGION)
        ac_pixels_sum = np.sum(image == cls.AC_DENSE_REGION)
        cc_pixels_sum = np.sum(image == cls.CC_DENSE_REGION)
        
        pda_cu  = (dense_pixels_sum+ac_pixels_sum+cc_pixels_sum)/(dense_pixels_sum+not_dense_pixels_sum+ac_pixels_sum+cc_pixels_sum)
        pda_ac = (ac_pixels_sum+cc_pixels_sum)/(dense_pixels_sum+not_dense_pixels_sum+ac_pixels_sum+cc_pixels_sum)
        pda_cc = (cc_pixels_sum)/(dense_pixels_sum+not_dense_pixels_sum+ac_pixels_sum+cc_pixels_sum)
        
        return (pda_cu, pda_ac, pda_cc)  
      
    @classmethod
    def get_multilevel_da(cls, image) :
        """ get dense areas (Cumulus, Altocumulus, Cirrocumulus) from segmented image
        (Cumulus, Altocumulus, Cirrocumulus) from segmented image
    
        Args:
            image (ndarray, [w,h,4]) : segmented image with label 0,1,2,3 
             0 : nondense breast region 
             1 : dense region
             2 : higher dense region
             3 : highest dense region
        Returns:
            float: dense area (pixel)
        """
        not_dense_pixels_sum = np.sum(image == cls.BREAST_REGION)
        dense_pixels_sum = np.sum(image == cls.DENSE_REGION)
        ac_pixels_sum = np.sum(image == cls.AC_DENSE_REGION)
        cc_pixels_sum = np.sum(image == cls.CC_DENSE_REGION)

        da_cu = (dense_pixels_sum+ac_pixels_sum+cc_pixels_sum)
        da_ac = (ac_pixels_sum+cc_pixels_sum)
        da_cc = (cc_pixels_sum)
                                                                       
        return [da_cu, da_ac, da_cc]
      
    @classmethod
    def get_multilevel_das(cls, files) : 
        return list(map(cls.get_multilevel_da, files))
    
    @classmethod
    def pixel_to_cm2(cls, das, width, height, manufacturer):
        if manufacturer == "GE": SCALE_FACTOR = cls.GE_SCALE_FACTOR
        elif manufacturer == "HOLOGIC": SCALE_FACTOR = cls.HOLOGIC_SCALE_FACTOR
          
        if(type(das) == np.int64): 
          return das*SCALE_FACTOR/float(width*height)
        else : 
          return [x*SCALE_FACTOR/float(width*height) for x in das]
      
    @classmethod
    @dispatch(type, object, (int, float), (int, float), (int, float), (int, float))
    def get_pda_by_threshold(cls, file, bg_threshold, dense_threshold, width, height) : 
        dicom = cls.load_dicom(file)
        image = cls.get_dicom_pixel(dicom)
        resized_image = cls.resize_image(image, width, height)
        return cls.get_pda_by_threshold(resized_image, bg_threshold, dense_threshold)
    
    @classmethod
    @dispatch(type, np.ndarray, (int, float), (int, float))
    def get_pda_by_threshold(cls, resized_image, bg_threshold, dense_threshold) : 
        """ get pda from thresholds
    
        Args:
            resized_image (ndarray) 
            bg_threshold (int, float): background threshold
            dense_threshold (int, float): dense region threshold 
        Returns:
            float: percent density (%)
        """
        dense = cls.get_dense_region(resized_image.copy(), threshold=dense_threshold)
        masked_image = cls.mask_air_region(resized_image.copy(), resized_image.copy(), background_threshold=bg_threshold)  
        air_region = cls.get_air_region(image=masked_image, threshold=bg_threshold) # Air region = 1, breast region = 0

        result = resized_image.copy()
        result[True == air_region] = cls.BACKGROUND_REGION
        result[True == dense]  = cls.DENSE_REGION
        result[(True != air_region) & (True != dense)] = cls.BREAST_REGION
          
        return cls.get_pda(result)
    
    @classmethod
    def breast_region_mask(cls, image, threshold, fill_breast_structures=False):
        binary_image = np.array(image > threshold, dtype=np.int8)+1

        # label image
        labels = measure.label(binary_image)

        background_label = 3 # I don't know why...

        #Fill the air around the person
        binary_image[background_label == labels] = 2

        if fill_breast_structures:
            # For every slice we determine the largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = cls.largest_label_volume(labeling, bg=0)

                if l_max is not None: #This slice contains some lung
                    binary_image[i][labeling != l_max] = 1

        # 1: breast 0: background
        binary_image -= 1 #Make the image actual binary

        # 0: brast 1: background
        binary_image = 1-binary_image # Invert it, lungs are now 1

        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max = cls.largest_label_volume(labels, bg=0)

        if l_max is not None: # There are air pockets
            binary_image[labels != l_max] = 0

        return binary_image
    
    @classmethod
    def largest_label_volume(cls, im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None

    @classmethod
    def mask_air_region(cls, image, result_image, background_threshold) :
        """ get air region and mask to result_image
    
        Args:
            image (ndarray): image for air region detection
            result_image (ndarray): image for air region masking
            background_threshold: threshold for air region detection
        Returns:
            ndarray: masked image with label -1
        """
        breast_fill = cls.breast_region_mask(image, fill_breast_structures=True, threshold=background_threshold)
        result_image[True == breast_fill] = cls.BACKGROUND_REGION
        return result_image

    def get_segmented_image(self, image, bg_threshold, dense_threshold) : 
        """ get segmented image with thresholds 
    
        Args:
            image (ndarray): 
            width (int or float)
            height (int or float)
            bg_threshold (int or float)
            dense_threshold (int or float)
        Returns:
            ndarray: segmented image
        """
        resized_image = self.resize_image(image)
        dense = self.get_dense_region(resized_image.copy(), threshold=dense_threshold)
        masked_image = self.mask_air_region(resized_image.copy(), resized_image.copy(), background_threshold=bg_threshold)  
        air_region = self.get_air_region(image=masked_image, threshold=bg_threshold) 
        
        result = resized_image.copy()
        result[True == dense]  = 1
        result[True == air_region] = -1
        result[(True != air_region) & (True != dense)] = 0
          
        return result
      
    def get_segmented_image_by_file(self, file, bg_threshold, dense_threshold) : 
        """ get segmented image with thresholds 
    
        Args:
            file (str) 
            width (int or float)
            height (int or float)
            bg_threshold (int or float)
            dense_threshold (int or float)
        Returns:
            ndarray: segmented image
        """
        dicom = self.load_dicom(file)
        image = self.get_dicom_pixel(dicom)
        result = self.get_segmented_image(image, bg_threshold, dense_threshold)
          
        return result
      
    def get_segmented_image_multiclass(self, file, bg_threshold, dense_threshold, ac_threshold, cc_threshold) : 
        """ get segmented image with three density thresholds 
    
        Args:
            file (str) 
            bg_threshold (int or float)
            dense_threshold (int or float)
            ac_threshold (int or float)
            cc_threshold (int or float)
        Returns:
            ndarray: segmented image
        """
        dicom = self.load_dicom(file)
        image = self.get_dicom_pixel(dicom)
        resized_image = self.resize_image(image)

        masked_image = self.mask_air_region(resized_image.copy(), resized_image.copy(), background_threshold=bg_threshold)  
        dense = self.get_dense_region(masked_image.copy(), threshold=dense_threshold)
        ac = self.get_dense_region(masked_image.copy(), threshold=ac_threshold)
        cc = self.get_dense_region(masked_image.copy(), threshold=cc_threshold)
        air_region = self.get_air_region(image=masked_image, threshold=bg_threshold) 
        
        result = masked_image.copy()
        result[True == dense]  = self.DENSE_REGION
        result[True == ac]  = self.AC_DENSE_REGION
        result[True == cc]  = self.CC_DENSE_REGION
        result[True == air_region] = self.BACKGROUND_REGION
        result[(True != air_region) & (True != dense)] = self.BREAST_REGION
          
        return result
    
    @classmethod
    def get_dense_region(cls, image, threshold) : 
        """ get binary image with threshold
        Args:
            image (ndarray) 
            threshold (int or float)
        Returns:
            ndarray: binary image  
            - 1: dense region 
            - 0: non-dense region 
        """
        density_image = np.array(image > threshold, dtype=np.int8)
        return density_image

    @classmethod
    def get_air_region(cls, image, threshold) : 
        """ get binary image with threshold
        Args:
            image (ndarray) 
            threshold (int or float)
        Returns:
            ndarray: binary image 
            - 1: air region
            - 0: breast region
        """
        bin_image = np.array(image < threshold, dtype=np.int8)
        return bin_image
         
    @classmethod
    @dispatch(type, np.ndarray)
    def get_bg_threhold_by_manufacturer(cls, image, width=256, height=224, manufacturer="GE") : 
        """ get background threshold by manufacturer
        Args:
            image (ndarray) 
            width (int or float)
            height (int or float)
            manufacturer (str)
            - GE
            - HOLOGIC
        Returns:
            float: background threshold
        """
        if manufacturer == "GE" : 
            threshold = cls.get_gmm_bg_threshold(image)
        elif manufacturer == "HOLOGIC" : 
            threshold = cls.HOLOGIC_THRESHOLD
        else : 
            threshold = cls.HOLOGIC_THRESHOLD
            raise Exception("No Manufacturer")
        return threshold 
      
    @classmethod
    @dispatch(type, object)
    def get_bg_threhold_by_manufacturer(cls, file, width=256, height=224) : 
        """ get background threshold by manufacturer
        Args:
            file (str) 
            width (int or float)
            height (int or float)
        Returns:
            float: background threshold
        """
        dicom = cls.load_dicom(file)
        manufacturer = DicomManager.get_manufacturer(dicom)
        image = cls.get_image(file, width, height)
        return cls.get_bg_threhold_by_manufacturer(image, width=width, height=height, manufacturer=manufacturer)
      
    @classmethod
    def get_gmm_bg_threshold(cls, image) : 
        """ get background threshold using gaussian mixture model
        Args:
            image (ndarray) 
            width (int or float)
            height (int or float)
        Returns:
            float: background threshold
        """
        classif = GaussianMixture(n_components=2)
        classif.fit(image.reshape([image.shape[0]*image.shape[1], 1]))

        threshold = np.mean(classif.means_)
        return threshold
    
    @classmethod
    def get_threshold_by_density(cls, image, bg_threshold, density, step_size) : 
        """ get dense threshold by density 
        Args:
            image (ndarray) 
            bg_threshold (int or float)
            density (int or float)
            step_size (int)
        Returns:
            float: dense threshold
        """
        prev_threshold = 0
        prev_density = 100
        max_value = np.max(image)

        for i, _ in enumerate(range(int(bg_threshold), int(max_value))) :
            if i == 0 :  
                dense_threshold = bg_threshold + step_size
            else : 
                dense_threshold = prev_threshold + step_size

            current_density = cls.get_pda_by_threshold(image, bg_threshold, dense_threshold) 
            current_diff = density - current_density
            prev_diff = density - prev_density

            if(current_diff * prev_diff < 0) : 
                step_size = step_size * -0.5

            if abs(step_size) < 1 : 
                return prev_threshold
            else : 
                prev_threshold = dense_threshold
                prev_density = current_density
    
    @classmethod
    def get_horizontal_vertical_flips(cls, x) : 
        """ get horizontal and vertical flips of images
        Args:
            x (ndarray): 4-D tensor 
        Returns:
            ndarray: augmented image tensor (4-D)
        """
        SAMPLE = x.shape[0]
        WIDTH = x.shape[1]
        HEIGHT = x.shape[2]
        CH = x.shape[3]

        augmented_x = np.zeros(shape=[4*SAMPLE,WIDTH,HEIGHT,CH])
        session = tf.Session()

        if tf.executing_eagerly() : 
          print("Eager mode")
          for i in tqdm(range(0, SAMPLE)):
              fliped_image = tf.image.flip_left_right(x[i]).numpy()
              fliped_image2 = tf.image.flip_up_down(x[i]).numpy()
              fliped_image3 = tf.image.flip_up_down(fliped_image).numpy()

              augmented_x[4*i] = x[i] # original image
              augmented_x[4*i+1] = fliped_image # vertical fliped image
              augmented_x[4*i+2] = fliped_image2 # horizontal fliped image
              augmented_x[4*i+3] = fliped_image3 # vertical and horizontal fliped image
        else : 
          for i in tqdm(range(0, SAMPLE)): 
              fliped_image = session.run(tf.image.flip_left_right(x[i]))
              fliped_image2 = session.run(tf.image.flip_up_down(x[i]))
              fliped_image3 = session.run(tf.image.flip_up_down(fliped_image))

              augmented_x[4*i] = x[i]
              augmented_x[4*i+1] = fliped_image
              augmented_x[4*i+2] = fliped_image2 
              augmented_x[4*i+3] = fliped_image3

        return augmented_x
      
    def make_preprocessed_images(self, files, normalizer): 
        """ make dicoms tensors
        Args:
            files (list): list of dicom files 
            normalizer (Normalizer): subclass of .lib.normalization.Normalizer
        Returns:
            ndarray: image tensor (4-D)
        """
        print(normalizer.print_info())
        print("--------------------------------------------------------------")
        print("Preparing images")
        result = np.zeros(shape=[len(files), self.width, self.height, 1])

        for index, file in enumerate(files) : 
            dicom = self.load_dicom(file)
            image = self.get_dicom_pixel(dicom)
            resized_image = self.resize_image(image) 
            bg_threshold = self.get_bg_threhold_by_manufacturer(file)
            masked_image = self.mask_air_region(resized_image.copy(), resized_image.copy(), background_threshold=bg_threshold) 
            normalized_image = normalizer.normalize(masked_image)
            result[index] = normalized_image.reshape([self.width, self.height, 1])

            if (index % 100 == 0):
                print("Processing image: ", index + 1, ", ", file)
                
        return result
    
    def make_ground_truths(self, files, thresholds):
        """ make dicoms ground truth
        Args:
            files (list): list of dicom files 
            thresholds (array-like): dense thresholds
        Returns:
            ndarray: 4D (sample_size*width*height*1)
        """
        print("Preparing images")

        result = np.zeros(shape=[len(files), self.width, self.height, 1])

        for index, file in enumerate(files): 
            dicom = self.load_dicom(file)
            image = self.get_dicom_pixel(dicom)
            resized_image = self.resize_image(image) 
            bg_threshold = self.get_bg_threhold_by_manufacturer(dicom, width=self.width, height=self.height)
            masked_image = self.mask_air_region(resized_image.copy(), resized_image.copy(), background_threshold=bg_threshold) 
            dense_threshold = thresholds[index]
            segmented_image = self.get_segmented_image(masked_image, bg_threshold=bg_threshold, 
                                                               dense_threshold=dense_threshold)

            result[index] = segmented_image.reshape([self.width, self.height, 1])

            if (index % 100 == 0):
                print("Processing image: ", index + 1, ", ", file)

        return result
      
    def make_ground_truths_multiclass(self, files, thresholds1, thresholds2, thresholds3):
        """ make dicoms ground truth
        Args:
            files (list): list of dicom files 
            thresholds1 (array-like): dense thresholds
            thresholds2 (array-like): higher thresholds
            thresholds3 (array-like): highest thresholds
        Returns:
            ndarray: 4D (sample_size*width*height*4)
        """
        print("Preparing images")

        result = np.zeros(shape=[len(files), self.width, self.height, 1])

        for index, file in enumerate(files): 
            dicom = self.load_dicom(file)
            # bg_threshold = self.get_bg_threhold_by_manufacturer(dicom, width=self.width, height=self.height)
            segmented_image = self.get_segmented_image_multiclass(file=file, 
                                                           bg_threshold=-1, 
                                                           dense_threshold=thresholds1[index], 
                                                           ac_threshold=thresholds2[index], 
                                                           cc_threshold=thresholds3[index])

            result[index] = segmented_image.reshape([self.width, self.height, 1])

            if (index % 100 == 0):
                print("Processing image: ", index + 1, ", ", file)

        return result
      
    
    @classmethod
    def one_hot_encoding(cls, tensor) : 
      """
      Args:
          tensor (ndarray): 4-D tensor 
      Returns:
          ndarray: one-hot-encoded tensor (4-D tensor)
      """
      tensor = tensor.astype(int)
      return np.eye(4)[tensor[:, :, :, 0]]
    
    @classmethod    
    def prob_map_to_classa_all(self, prob_map) : 
        """ opposite of one_hot_encoding
        Args:
            prob_map (ndarray): 4-D tensor 
        Returns: 
            ndarray: last axis value means argmax density classes (4-D tensor)
        """
        return np.argmax(prob_map[:,:,:,:], axis=3)
      
    @classmethod    
    def prob_map_to_class(self, prob_map) : 
        """ opposite of one_hot_encoding
        Args:
            prob_map (ndarray): 3-D tensor 
        Returns: 
            ndarray: last axis value means argmax density classes (3-D tensor)
        """
        return np.argmax(prob_map[:,:,:], axis=2)
      
    @classmethod
    def get_dense_region_between_thresholds(cls, image, threshold1, threshold2) : 
        """ get binary image with two thresholds
        Args:
            image (ndarray) 
            threshold1 (int or float)
            threshold2 (int or float)
        Returns:
            ndarray: binary image  
            - 1: interest region
            - 0: non-dense region 
        """
        density_image = np.array((image > threshold1) & (image < threshold2), dtype=np.int8)
        return density_image