import pydicom
import numpy as np
import pandas as pd 
import os
from skimage import measure, morphology
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
import pydicom
import tensorflow as tf
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from multipledispatch import dispatch
import sys
import time
from keras.preprocessing.image import ImageDataGenerator
from mammo2risk.dicom_manager import DicomManager
from mammo2risk.preprocessing import Preprocessor

import warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn

class Visualizer(object) :  
    def __init__(self, width, height) : 
        self.width = width
        self.height = height

    @classmethod
    @dispatch(type, np.ndarray)
    def gray_plot(cls, image, file=False) : 
        fig = plt.figure()
        ax = plt.imshow(image, cmap=plt.cm.gray)
        ax = plt.Axes(fig,[0,0,1,1])
        plt.axis('off')
        
        if (file != False) : 
          plt.savefig(file, transparent = True, bbox_inches = 'tight', pad_inches = 0)
        
        plt.show()
        
    @classmethod
    @dispatch(type, object)
    def gray_plot(cls, path, file=False) : 
        image = Preprocessor.get_image(path)
        cls.gray_plot(image, file=file)
        
    @classmethod
    @dispatch(type, object)
    def plot_histogram(cls, path, title="Histogram", ylim=False, hist=True, xlim1=False, xlim2=False, file=False) : 
        dicom = pydicom.read_file(path, force=True)
        image = dicom.pixel_array 
        cls.plot_histogram(image) 

    @classmethod
    @dispatch(type, np.ndarray)
    def plot_histogram(cls, img, title="Histogram", ylim=False, hist=True, xlim1=False, xlim2=False, file=False, threshold=False) :
        if threshold:
            img = img[img > threshold]
        histogram = img.flatten()
        fig, ax = plt.subplots()
        fig.set_size_inches(11.7, 8.27)
        sns.distplot(histogram, color='black', hist=hist, bins=55)
        plt.xlabel("Pixel intensity", size=17)
        plt.ylabel("Probability density", size=17)
        plt.yticks(fontsize = 15)
        plt.xticks(fontsize = 15)
        
        if xlim1 != False : 
            plt.xlim(xlim1, xlim2)
        if ylim != False : 
            plt.ylim(0, ylim)
        plt.title(title, size=17)
        
        if (file != False) : 
            plt.savefig(file, transparent = True, bbox_inches = 'tight', pad_inches = 0)
        plt.show()
    
    @classmethod
    def plot_normalized_histogram(cls, img, normalizer, title="Histogram", ylim=False, hist=True, xlim1=False, xlim2=False, file=False) :
        normalized_image = normalizer.normalize(img)
        cls.plot_histogram(normalized_image, xlim1=xlim1, xlim2=xlim2, ylim=ylim, title=title, file=file)
    