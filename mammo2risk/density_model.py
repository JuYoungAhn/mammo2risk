import numpy as np
import cv2 
from skimage import measure, morphology
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import seaborn as sns
import sys
from abc import ABCMeta, abstractmethod
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import Sequential
from keras import backend as K
from multipledispatch import dispatch
import os

import warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn

from mammo2risk.metric import weighted_categorical_crossentropy
from mammo2risk.metric import mean_iou
from mammo2risk.metric import gen_dice_loss
from mammo2risk.dicom_manager import DicomManager
from mammo2risk.monitoring import CorrelationMonitoring
from mammo2risk.unet import *

class DensityModelInterface(metaclass=ABCMeta): 
      BACKGROUND_REGION = -1
      BREAST_REGION = 0
      DENSE_REGION = 1
      AC_DENSE_REGION = 2
      CC_DENSE_REGION = 3
      HOLOGIC_THRESHOLD = 50
      ERROR_VALUE = 999
      
      @abstractmethod
      def get_pda(self, input_file, skin): 
          pass
      @abstractmethod
      def get_da(self, input_file, skin): 
          pass
      @abstractmethod
      def get_pdas(self, input_files, skin): 
          pass
      @abstractmethod
      def get_das(self, input_files, skin):
          pass
      @abstractmethod
      def get_prob_map(self, input_file):
          pass
      @abstractmethod
      def get_prob_maps(self, input_files): 
          pass
      @abstractmethod
      def get_segmented_image(self, input_file):
          pass
      @abstractmethod
      def train(self): 
          pass

class DeepDensity(DensityModelInterface) : 
    def __init__(self, normalizer, preprocessor) : 
        self.normalizer = normalizer
        self.preprocessor = preprocessor
        
    @property
    def preprocessor(self):
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, preprocessor):
        self._preprocessor = preprocessor   
   
    @property
    def normalizer(self):
        return self._normalizer

    @normalizer.setter
    def normalizer(self, normalizer):
        self._normalizer = normalizer 
     
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_path):
        model = UNet(img_shape=(self._preprocessor.width, self._preprocessor.height, 1), batchnorm=True)
        model.load_weights(model_path)
        self._model = model
        self._model_path = model_path

    def export_model(self, path): 
        if self.model:
          self.model.save_weights(path)
        else:
          raise("Nothing to export !!")
          
    def train(self, train_x, train_y, valid_x, valid_y, epoch, out_act, loss, 
              optimizer, metrics, batch_size, callbacks): 
        """ train model

        Args:
            train_x (ndarray)
            train_y (ndarray)
            valid_x (ndarray)
            valid_y (ndarray)
            epoch (int)
            loss (function)
            optimizer (list of keras.optimizers)
            metrics (function)
            batch_size (int)
            callback (list of keras.callbacks)
            export_path (str)
        Returns:
            history
        """
        print(f"{type(self).__name__} Training Start")

        model = UNet(img_shape=(self._preprocessor.width, self._preprocessor.height, 1), 
                     out_ch=train_y.shape[3], batchnorm=True, out_act=out_act)
        
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        hist = model.fit(
            train_x,
            train_y,
            nb_epoch=epoch,
            batch_size=batch_size,
            verbose=2,
            validation_data=[valid_x, valid_y],
            callbacks=callbacks
        )
        
        self._model = model
        
        return hist
  
    def _get_class_prediction(self, prob_map, threshold) : 
        """ private method for thresholding

        Args:
            prob_map (ndarray)
            threshold (float)
        Returns:
            ndarray: binary map
        """
        prob_map = np.where(prob_map >= threshold, 1, prob_map)
        prob_map = np.where(prob_map < threshold, 0, prob_map)
        return prob_map
    
    def get_pda(self, input_file, skin, prob_threshold) : 
        """ get percent density

        Args:
            input_file (str)
            skin (boolean): whether exclude skin or not
            prob_threshold (float): threshold to segmented dense region
        Returns:
            float: percent density
        """
        dicom = self._preprocessor.load_dicom(input_file) # Load dicom from file path
        image = self._preprocessor.get_dicom_pixel(dicom) # get pixel from dicom
        resized_image = self._preprocessor.resize_image(image) # resize image
        threshold = self._preprocessor.get_bg_threhold_by_manufacturer(dicom) 
        
        air_region = self.get_air_region(image=resized_image, threshold=threshold) # background(air) masking Air region = 1, breast region = 0
        skin_region = self.skin_clustering(air_region, width=200) # Skin region = 1, others = 0
        normalized_image = self._normalizer.normalize(resized_image)

        pred = self._model.predict(normalized_image.reshape([1, self._preprocessor.width, self._preprocessor.height, 1])) # Dense region prediction
        pred = self._get_class_prediction(pred, prob_threshold) # covert probability map to class map
     
        segmented_image = pred.copy() # Final segmentation map 1 : Dense region 0 : Non-dense region -1 : Air region
        segmented_image = segmented_image.reshape(self._preprocessor.width, self._preprocessor.height, 1)
        segmented_image[True == air_region] = self.BACKGROUND_REGION

        if skin : 
            segmented_image[True == skin_region] = self.BREAST_REGION

        return self._preprocessor.get_pda(segmented_image)
      
    def get_da(self, input_file, skin, prob_threshold) : 
        dicom = self._preprocessor.load_dicom(input_file) # Load dicom from file path
        image = self._preprocessor.get_dicom_pixel(dicom) # get pixel from dicom
        resized_image = self._preprocessor.resize_image(image) # resize image
        threshold = self._preprocessor.get_bg_threhold_by_manufacturer(dicom) 
        
        air_region = self.get_air_region(image=resized_image, threshold=threshold) # background(air) masking Air region = 1, breast region = 0
        skin_region = self.skin_clustering(air_region, width=200) # Skin region = 1, others = 0
        normalized_image = self._normalizer.normalize(resized_image)

        pred = self._model.predict(normalized_image.reshape([1, self._preprocessor.width, self._preprocessor.height, 1])) # Dense region prediction
        pred = self._get_class_prediction(pred, prob_threshold) # covert probability map to class map
        pred = pred.reshape([self._preprocessor.width, self._preprocessor.height])
        
        segmented_image = pred.copy() # Final segmentation map 1 : Dense region 0 : Non-dense region -1 : Air region
        segmented_image[True == air_region] = self.BACKGROUND_REGION

        if skin : 
            segmented_image[True == skin_region] = BREAST_REGION

        return self._preprocessor.get_da(segmented_image)
      
    def get_pdas(self, input_files, skin, prob_threshold) :  
        pdas = []
        for file in tqdm(input_files): 
            try : 
                pda = self.get_pda(input_file=file, skin=skin, prob_threshold=prob_threshold)
                pdas.append(pda)
            except Exception as e: 
                print(e)
                pdas.append(self.ERROR_VALUE)
        return pdas
      
    def get_das(self, input_files, skin, prob_threshold) :  
        pdas = []
        for file in tqdm(input_files): 
            try : 
                pda = self.get_da(input_file=file, skin=skin, prob_threshold=prob_threshold)
                pdas.append(pda)
            except Exception as e: 
                print(e)
                pdas.append(ERROR_VALUE)
        return pdas

    def skin_clustering(self, image, width=255) : 
        INPUT = image.copy()
        MASK = np.array(INPUT, dtype='float32')

        MASK = cv2.GaussianBlur(MASK, (25,25), 300)
        BG = np.ones([INPUT.shape[0], INPUT.shape[1], 1], dtype='uint8')*255
        OUT_F = np.ones([INPUT.shape[0], INPUT.shape[1], 1],dtype='uint8')

        for r in range(INPUT.shape[0]):
            for c in range(INPUT.shape[1]):
                OUT_F[r][c] = int(BG[r][c]*(MASK[r][c]) + INPUT[r][c]*(1-MASK[r][c]))

        outimage = OUT_F
        skin_image = outimage.copy()

        skin_image[np.where((0 < skin_image) & (skin_image <= width))] = 100 # Blurring 된 부분을 100으로 바꿈
        skin_image[np.where(skin_image != 100)] = 0
        skin_image[np.where(skin_image == 100)] = 1 
        skin_image [True == INPUT] = 0 # Blurring 된 부분 중, Breast area와 겹치는 부분만 골라 냄
        return skin_image.reshape([self._preprocessor.width, self._preprocessor.height])
    
    def breast_region_mask(self, image, threshold, fill_breast_structures=False):
        # Air를 분리하고 breast 부분만 찾음 (Denoising)
        # not actually binary, but 1 and 2. 
        # 0 is treated as background, which we do not want 
        # 2 : breast
        # 1 : background
        binary_image = np.array(image > threshold, dtype=np.int8)+1

        # image에 label을 부여함
        labels = measure.label(binary_image)

        background_label = 3 # I don't know why

        #Fill the air around the person
        binary_image[background_label == labels] = 2

        if fill_breast_structures:
            # For every slice we determine the largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = self.largest_label_volume(labeling, bg=0)

                if l_max is not None: #This slice contains some lung
                    binary_image[i][labeling != l_max] = 1

        # 1: breast 0: background
        binary_image -= 1 #Make the image actual binary

        # 0: brast 1: background
        binary_image = 1-binary_image # Invert it, lungs are now 1

        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max = self.largest_label_volume(labels, bg=0)

        if l_max is not None: # There are air pockets
            binary_image[labels != l_max] = 0

        return binary_image
    
    # 가장 volume이 큰 label을 찾는다
    def largest_label_volume(self, im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None

    # background threshold를 통해 air region을 labeling함
    def mask_air_region(self, image, result_image, background_threshold) :
        breast_fill = self.breast_region_mask(image, fill_breast_structures=True, threshold=background_threshold)
        result_image[True == breast_fill] = self.BACKGROUND_REGION
        return result_image
    
    # threshold 아래를 0, 이외는 1을 반환
    def get_air_region(self, image, threshold) : 
        bin_image = np.array(image < threshold, dtype=np.int8)
        return bin_image
    
    def get_segmented_image(self, input_file, prob_threshold=0.5, skin=False) : 
        dicom = self._preprocessor.load_dicom(input_file) # Load dicom from file path
        image = self._preprocessor.get_dicom_pixel(dicom) # get pixel from dicom
        resized_image = self._preprocessor.resize_image(image) # resize image
        threshold = self._preprocessor.get_bg_threhold_by_manufacturer(dicom) 
        
        # background (air) masking 
        masked_image = self.mask_air_region(resized_image.copy(), resized_image.copy(), background_threshold=threshold) 
        normalized_image = self._normalizer.normalize(resized_image)
        

        # Dense region prediction
        pred = self._model.predict(normalized_image.reshape([1, self._preprocessor.width, self._preprocessor.height, 1]))
        pred = self._get_class_prediction(pred, prob_threshold) 
        pred = pred.reshape([self._preprocessor.width, self._preprocessor.height])

        air_region = self.get_air_region(image=masked_image, threshold=threshold) # Air region = 1, breast region = 0
        skin_region = self.skin_clustering(air_region, width=200) # Skin region = 1, others = 0

        # Final segmentation map 1 : Dense region, 0 : Non-dense region, -1 : Air region
        segmented_image = pred.copy()
        segmented_image[True == air_region] = self.BACKGROUND_REGION

        if skin : 
            segmented_image[True == skin_region] = self.BREAST_REGION

        return segmented_image
      
    def plot_segmentation_result(self, input_file, prob_threshold=0.5, skin=False, file=False) :
        sns.set_style("dark")

        cumulus_result = self.get_segmented_image(input_file, prob_threshold=prob_threshold, skin=skin)
        cumulus_result[cumulus_result == -1] = 0 

        dicom = self._preprocessor.load_dicom(input_file)
        image = self._preprocessor.get_dicom_pixel(dicom)
        resized_image = self._preprocessor.resize_image(image) # resize image
        threshold = self._preprocessor.get_bg_threhold_by_manufacturer(input_file) 
        
        masked_image = self.mask_air_region(resized_image.copy(), resized_image.copy(), background_threshold=threshold)  
        normalized_image = self._normalizer.normalize(masked_image)

        ax = plt.imshow(normalized_image.reshape([self._preprocessor.width, self._preprocessor.height]), cmap=plt.cm.gray)
        plt.imshow(cumulus_result, cmap="jet", alpha=0.5, interpolation='bilinear')
        # plt.imshow(cumulus_result, cmap="gnuplot2", alpha=0.4, interpolation='bilinear')
        plt.axis('off')
        
        if (file != False) : 
          plt.savefig(file, transparent = True, bbox_inches = 'tight', pad_inches = 0)
        
        plt.show()
        
    def save_image(self, input_file, save_path, save_root, prob_threshold=0.5, skin=False) :
        sns.set_style("dark")
        print("Saving Image..")
        cumulus_result = self.get_segmented_image(input_file, prob_threshold=prob_threshold, skin=skin)
        cumulus_result[cumulus_result == -1] = 0 
        
        dicom = self._preprocessor.load_dicom(input_file)
        image = self._preprocessor.get_dicom_pixel(dicom)
        resized_image = self._preprocessor.resize_image(image) # resize image
        threshold = self._preprocessor.get_bg_threhold_by_manufacturer(input_file) 
        masked_image = self.mask_air_region(resized_image.copy(), resized_image.copy(), background_threshold=threshold)  
        normalized_image = self._normalizer.normalize(masked_image)

        # equivalent but more general
        ax1 = plt.subplot(1, 2, 1)
        ax1 = plt.imshow(normalized_image.reshape([self._preprocessor.width, self._preprocessor.height]), cmap=plt.cm.gray)
        plt.axis('off')
        
        ax2 = plt.subplot(1, 2, 2)
        ax2 = plt.imshow(normalized_image.reshape([self._preprocessor.width, self._preprocessor.height]), cmap=plt.cm.gray)
        ax2 = plt.imshow(cumulus_result, cmap="jet", alpha=0.5, interpolation='bilinear')
        plt.axis('off')
        
        path, file = os.path.split(input_file)
        filename = os.path.relpath(input_file, save_root)
        filename = filename.replace("../", "")
        directory = os.path.dirname(save_path+'/'+filename)
        
        if not os.path.exists(directory):
            os.mkdir(directory)
        
        filename2, file_extension = os.path.splitext(filename)
        save_name = filename2+".jpg"
        
        save_path_final = save_path+"/"+save_name
        
        if not os.path.exists(save_path_final):
          save_path = os.path.abspath(save_path_final)
          plt.savefig(save_path, transparent = True, bbox_inches = 'tight', pad_inches = 0)
          
        return 0
    
class MultiDensity(DeepDensity) : 
    def __init__(self, normalizer, preprocessor) : 
        self.normalizer = normalizer
        self.preprocessor = preprocessor
   
    @property
    def model(self):
        return self._model
      
    @model.setter
    def model(self, model_path):
        model = UNet(img_shape=(self._preprocessor.width, self._preprocessor.height, 1), batchnorm=True, out_ch=4, out_act='softmax')
        model.load_weights(model_path)
        self._model = model
    
    def predict_prob(self, x) : 
        return self.model.predict(x)
    
    def predict_class(self, x) : 
        x = x.reshape(1, self._preprocessor.width, self._preprocessor.height, 1)
        prob_map = self.predict_prob(x) 
        result = np.argmax(prob_map[:,:,:,:], axis=3) 
        return result 
    
    def get_prob_map(self, file) : 
        dicom = self._preprocessor.load_dicom(file) # Load dicom from file path
        image = self._preprocessor.get_dicom_pixel(dicom) # get pixel from dicom
        resized_image = self._preprocessor.resize_image(image) # resize image
        normalized_image = self._normalizer.normalize(resized_image)
        prob_map = self.predict_prob(
          normalized_image.reshape(1, self._preprocessor.width, self._preprocessor.height, 1)) 

        return np.squeeze(prob_map)
    
    def get_prob_maps(self, files) : 
        if isinstance(files, str): result = self.get_prob_map(files)
        else :
          result = np.zeros(shape=[len(files), self._preprocessor.width, self._preprocessor.height, 4])
          for i, file in tqdm(enumerate(files)) : 
            result[i] = self.get_prob_map(file)
            
        return result
    
    def get_density_profile(self, file) : 
        dicom = self._preprocessor.load_dicom(file) # Load dicom from file path
        image = self._preprocessor.get_dicom_pixel(dicom) # get pixel from dicom
        resized_image = self._preprocessor.resize_image(image) # resize image
        normalized_image = self._normalizer.normalize(resized_image)
        prob_map = self.predict_prob(
          normalized_image.reshape(1, self._preprocessor.width, self._preprocessor.height, 1)) 
        

        prob_map = np.squeeze(prob_map)

        idx = np.where(
            (self._preprocessor.prob_map_to_class(prob_map) != self.BACKGROUND_REGION)
            & (self._preprocessor.prob_map_to_class(prob_map) != self.BREAST_REGION)
        )
        
        profile = prob_map[idx]
        profile = profile[:,1:4]
        
        return profile
    
    def get_density_profiles(self, files) : 
        if isinstance(files, str): result = self.get_density_profile(files)
        else :
          result = [0 for i in range(len(files))]
          for i, file in tqdm(enumerate(files)) : 
            result[i] = self.get_density_profile(file)
        return result
    
    def get_density_scores(self, profiles): 
        result = [0 for i in range(len(profiles))]
        for i, profile in enumerate(profiles): 
            cu_score = np.sum(profile[:,0])
            ac_score = np.sum(profile[:,1])
            cc_score = np.sum(profile[:,2])
            result[i] = (cu_score, ac_score, cc_score)
        return result
      
    def get_segmented_image(self, file, prob_threshold=0.5, skin=False) : 
        dicom = self._preprocessor.load_dicom(file) # Load dicom from file path
        image = self._preprocessor.get_dicom_pixel(dicom) # get pixel from dicom
        resized_image = self._preprocessor.resize_image(image) # resize image
        threshold = self._preprocessor.get_bg_threhold_by_manufacturer(file) 

        # background (air) masking 
        masked_image = self.mask_air_region(resized_image.copy(), resized_image.copy(), background_threshold=threshold) 
        normalized_image = self._normalizer.normalize(resized_image)
        
        # Dense region prediction
        prob_map = self.predict_prob(normalized_image.reshape(1, self._preprocessor.width, self._preprocessor.height, 1)) 
        result = np.argmax(prob_map[:,:,:,:], axis=3).reshape(self._preprocessor.width, self._preprocessor.height)

        air_region = self.get_air_region(image=masked_image, threshold=threshold) # Air region = 1, breast region = 0
        skin_region = self.skin_clustering(air_region, width=200) # Skin region = 1, others = 0

        # Final segmentation map 
        segmented_image = result.copy()
        segmented_image[True == air_region] = self.BACKGROUND_REGION

        if skin : 
            segmented_image[True == skin_region] = self.BREAST_REGION

        return segmented_image
    
    def get_multilevel_pda(self, file) : 
        try : 
          result = self.get_segmented_image(file)
          pda = self._preprocessor.get_multilevel_pda(result)
        except Exception as e: 
          print(e)
          pda = self.ERROR_VALUE
        return pda
    
    def get_multilevel_da(self, file) : 
        try : 
          result = self.get_segmented_image(file)
          da = self._preprocessor.get_multilevel_da(result)
        except Exception as e: 
          print(e)
          da = self.ERROR_VALUE
        return da
    
    def get_multilevel_pdas(self, files) : 
        return list(map(self.get_multilevel_pda, tqdm(files)))
    
    def get_multilevel_das(self, files) : 
        return list(map(self.get_multilevel_da, tqdm(files)))
      
    def get_pda(self, file) :
        return self.get_multilevel_pda(file)[0]
    
    def get_pdas(self, files) : 
        return list(map(self.get_pda, tqdm(files)))
    
    def get_pda_ac(self, file) :
        return self.get_multilevel_pda(file)[1]
    
    def get_pdas_ac(self, files) :  
        return list(map(self.get_pda_ac, tqdm(files)))
    
    def get_pda_cc(self, file) : 
        return self.get_multilevel_pda(file)[2]
      
    def get_pdas_cc(self, files) :  
        return list(map(self.get_pda_cc, tqdm(files)))
      
    def print_multi_densities(self, file): 
        das = self.get_multilevel_da(file)
        pdas = self.get_multilevel_pda(file)
        
        dicom = self.preprocessor.load_dicom(file)
        manufacturer = DicomManager.get_manufacturer(dicom)
        
        da_cu = self.preprocessor.pixel_to_cm2(das[0], self.preprocessor.width, self.preprocessor.height, manufacturer)
        da_ac = self.preprocessor.pixel_to_cm2(das[1], self.preprocessor.width, self.preprocessor.height, manufacturer)
        da_cc = self.preprocessor.pixel_to_cm2(das[2], self.preprocessor.width, self.preprocessor.height, manufacturer)
        
        pda_cu = pdas[0]*100
        pda_ac = pdas[1]*100
        pda_cc = pdas[2]*100
        
        print(f"Cumulus(cm2):{round(da_cu, 2)}")
        print(f"Altocumulus(cm2):{round(da_ac, 2)}")
        print(f"Cirrocumulus(cm2):{round(da_cc, 2)}")
        print(f"Cumulus(%):{round(pda_cu, 2)}")
        print(f"Altocumulus(%):{round(pda_ac, 2)}")
        print(f"Cirrocumulus(%):{round(pda_cc, 2)}")