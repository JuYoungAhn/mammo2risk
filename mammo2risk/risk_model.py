import numpy as np
import pandas as pd 
import sys
from tqdm import tqdm
from mammo2risk.preprocessing import Preprocessor
from mammo2risk.resnet import ResNet
from sklearn.preprocessing import StandardScaler
from keras.applications.resnet50 import preprocess_input
from sklearn.externals import joblib

import warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn

class DeepMammoRisk(object): 
    def __init__(self, preprocessor, normalizer) :
        self.preprocessor = preprocessor
        self.normalizer = normalizer
        return
      
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
        model = ResNet(top_fix=False, do=0.5, weights='imagenet')
        model.load_weights(model_path)
        self._model = model
        self._model_path = model_path
    
    def get_deep_mammo_score(self, file): 
        image = self.preprocessor.resize_image(file)
        norm_image = self.normalizer.normalize(image)
        norm_image = norm_image[..., None]
        norm_image = np.repeat(norm_image, 3, axis=2)
        norm_image = norm_image[None, ...]
        preprocessed_image = preprocess_input(norm_image)
        return self.model.predict_proba(preprocessed_image)[0][0]
      
    def get_deep_mammo_scores(self, files):
        return list(map(self.get_deep_mammo_score, tqdm(files)))

class DenseRisk(object) : 
    def __init__(self, preprocessor, normalizer) :
        self.preprocessor = preprocessor
        self.normalizer = normalizer
        return
      
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
    def num_feature(self):
        try :
          return self._num_feature
        except:
          print("Please set num_feature.")

    @num_feature.setter
    def num_feature(self, num_feature):
        self._num_feature = num_feature
    
    @property
    def risk_model(self):
        try :
          return self._risk_model
        except:
          print("Please set risk model.")

    @risk_model.setter
    def risk_model(self, model):
        self._risk_model = model
        
    @property
    def scaler(self):
        try :
          return self._scaler
        except:
          print("Please set scaler.")

    @scaler.setter
    def scaler(self, scaler):
        self._scaler = scaler    
        
    def get_density_score_from_features(self, features): 
        if features.ndim == 1: features.shape([1, len(features)])
        y_probas = self.risk_model.predict_proba(features)
        y_probas = y_probas[:, 1]
        return y_probas
    
    def get_density_score(self, file): 
        """ get deep density scores

        Args: 
          files (list) 

        Returns: 
          float: density score
        """
        image = self.preprocessor.get_image(file)
        norm_image = self.normalizer.normalize(image)
        feature = self.extract_features(norm_image, self._num_feature)
        feature = feature.reshape([1, self._num_feature])
        # Standardization
        scaler = joblib.load(self.scaler)
        std_density_feature = scaler.transform(feature)
        score = self.get_density_score_from_features(std_density_feature)[0]
        return score 
        
    def get_density_scores(self, files): 
        """ get deep density scores

        Args: 
          files (list) 

        Returns: 
          list: list of density scores
        """    
        norm_image = self.preprocessor.make_preprocessed_images(
              files=files, normalizer=self.normalizer
        )
        features = self.extract_features_all(norm_image, self._num_feature)
        scores = self.get_density_score_from_features(features)
        return scores

      
    @classmethod
    def extract_features(cls, prob_map, num_feature) :
        """ extract features from prob map of density model

        Args: 
          prob_map (ndarray) 
          num_threshold (int) 
        Returns: 
          ndarray ([num_threshold]): features
        """        
        features = np.zeros(shape=[num_feature])
        for i in range(0, num_feature) : 
            threshold = 1/num_feature*i
            threshold2 = threshold + 1/num_feature
            class_map = Preprocessor.get_dense_region_between_thresholds(prob_map, threshold, threshold2)
            da = Preprocessor.get_da(class_map)
            features[i] = da
        return features

    @classmethod
    def extract_features_all(cls, prob_maps, num_feature): 
        """ get standardized features from prob map of density model

        Args: 
          prob_maps (ndarray) 
          num_threshold (int) 
        Returns: 
          ndarray ([sample_size*num_threshold]): features
        """     
        features = np.zeros(shape=[prob_maps.shape[0], num_feature])
        for i in tqdm(range(prob_maps.shape[0])) :
            features[i,:] = cls.extract_features(prob_map=prob_maps[i], num_feature=num_feature)
        
        return features
      
    @classmethod
    def extract_naive_feature(cls, file, width, height, num_threshold): 
        image = Preprocessor.get_image(file, width, height)
        bg_threshold = Preprocessor.get_bg_threhold_by_manufacturer(file, width=width, height=height)
        
        step = (4095 - bg_threshold) / num_threshold
        
        features = np.zeros(shape=[num_threshold])
        for i in range(0, num_threshold) : 
            threshold1 = bg_threshold + step*i
            threshold2 = threshold1 + step
            class_map = Preprocessor.get_dense_region_between_thresholds(image, threshold1, threshold2)
            da = Preprocessor.get_da(class_map)
            features[i] = da
        return features
      
    @classmethod
    def extract_naive_feature_all(cls, files, width, height, num_threshold): 
        result = np.zeros(shape=[len(files), num_threshold])
        for i,file in tqdm(enumerate(files)) :
            result[i,:] = cls.extract_naive_feature(file, width, height, num_threshold)
        
        scaler = StandardScaler() 
        std_density_feature = scaler.fit_transform(result) 
        return std_density_feature
      
class DeepDenseRisk(object): 
    def __init__(self, preprocessor, normalizer) :
        self.preprocessor = preprocessor
        self.normalizer = normalizer
        return
    
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
    def num_feature(self):
        try :
          return self._num_feature
        except:
          print("Please set num_feature.")

    @num_feature.setter
    def num_feature(self, num_feature):
        self._num_feature = num_feature
    
    @property
    def density_model(self):
        try :
          return self._density_model
        except:
          print("Please set density model.")

    @density_model.setter
    def density_model(self, model):
        model.print_config()
        self._density_model = model
      
    @property
    def risk_model(self):
        try :
          return self._risk_model
        except:
          print("Please set risk model.")

    @risk_model.setter
    def risk_model(self, model):
        self._risk_model = model
        
    @classmethod
    def extract_features_multiclass(cls, prob_map, num_feature) :
        """ extract features from prob map of density model

        Args: 
          prob_map (ndarray) 
          num_threshold (int) 
        Returns: 
          ndarray ([num_feature,3]): features 
        """      
        cu_prob = prob_map[..., 1]
        ac_prob = prob_map[..., 2]
        cc_prob = prob_map[..., 3]

        features_cu = cls.extract_features(cu_prob, num_feature)
        features_ac = cls.extract_features(ac_prob, num_feature)
        features_cc = cls.extract_features(cc_prob, num_feature)

        result = np.concatenate((features_cu, features_ac, features_cc))
        return result
      
    @classmethod
    def extract_features_multiclass_all(cls, prob_maps, num_feature) : 
        """ extract features from prob map of density model

        Args: 
          prob_maps (ndarray) 
          num_threshold (int) 
        Returns: 
          ndarray ([sample_size*num_feature,3]): features
        """
        if(prob_maps.ndim == 3): prob_maps = np.expand_dims(prob_maps, axis=0)
        result = np.zeros(shape=[prob_maps.shape[0], (num_feature)*3])
           
        for i in tqdm(range(prob_maps.shape[0])) :
            result[i,:] = cls.extract_features_multiclass_for_prob_map(prob_map=prob_maps[i], num_feature=num_feature)
        
        scaler = StandardScaler() 
        std_density_feature = scaler.fit_transform(result) 
        return std_density_feature  
      
    def get_deep_density_score(self, file):
        """ get deep density score 

        Args: 
          file (str) 

        Returns: 
          float: density score
        """
        prob_map = self.density_model.get_prob_map(image)
        features = self.extract_features_multiclass_for_prob_map(prob_map, self._num_feature)
        density_score = self.get_density_score_from_features(features)
        return density_score
        
    def get_deep_density_scores(self, files) :
        """ get deep density scores

        Args: 
          files (list) 

        Returns: 
          list: list of density scores
        """
        prob_maps = self.density_model.get_prob_maps(files)
        features = self.extract_features_multiclass_for_prob_maps(prob_maps, self._num_feature)
        density_scores = self.get_density_score_from_features(features)
        return density_scores