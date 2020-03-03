import cv2
import numpy as np
from skimage.transform import resize
from abc import ABCMeta, abstractmethod
import sys

class Normalizer(metaclass=ABCMeta) : 
    def print_info(self) : 
        print(type(self).__name__)
        print(self._param)
        
    def __init__(self, **param) :
        self._param = param

    @abstractmethod
    def normalize(self, image) :
        pass
        
class CLAHENormalizer(Normalizer) :    
    def normalize(self, image) :
        result = self._CLAHE(image, **self._param)
        result = result.reshape([image.shape[0], image.shape[1]])
        
        return result
    
    def _CLAHE(self, image, max_value, norm_max_value, grid_size, cliplimit) : 
        img = image / max_value * 255
        img = img.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(grid_size,grid_size))
        result = clahe.apply(img)
        result = result / 255 * norm_max_value

        return result
        