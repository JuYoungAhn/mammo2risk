import sys
from mammo2risk.density_model import DeepDensity
from mammo2risk.density_model import MultiDensity
from mammo2risk.preprocessing import Preprocessor
from mammo2risk.visualizer import Visualizer
from mammo2risk.normalization import CLAHENormalizer
from mammo2risk.dicom_manager import DicomManager
from mammo2risk.risk_model import DenseRisk
from mammo2risk.risk_model import DeepMammoRisk
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import json
from pathlib import Path

class MammoRiskManager(object): 
    GE_MD_NORMALIZER_CONFIG = {"max_value": 4095, "norm_max_value": 1, "grid_size": 4, "cliplimit": 4}
    HO_MD_NORMALIZER_CONFIG = {"max_value": 4095, "norm_max_value": 1, "grid_size": 4, "cliplimit": 1}
    GE_MR_NORMALIZER_CONFIG = {"max_value": 4095, "norm_max_value": 255, "grid_size": 4, "cliplimit": 4}
    HO_MR_NORMALIZER_CONFIG = {"max_value": 4095, "norm_max_value": 255, "grid_size": 4, "cliplimit": 1}
    def __init__(self, ge_md, ho_md, ge_mr, ho_mr):
      self.GE_MD = ge_md
      self.HO_MD = ho_md
      self.GE_MR = ge_mr
      self.HO_MR = ho_mr
      #self.GE_DR_COEF = ge_dr_coef
      #self.GE_DR_INT = ge_dr_int
      #self.HO_DR_COEF = ho_dr_coef
      #self.HO_DR_INT = ho_dr_int
      #self.GE_DR_SCALER = ge_dr_scaler
      #self.HO_DR_SCALER = ho_dr_scaler
      #self.DR_NUM_FEATURES = num_features
      return
    
    @classmethod
    def get_config(cls, path): 
      print(f"Config path {path}")
      weight_path = str(Path(path).parent)
      config_path = path

      with open(config_path, "r") as f:
          weights = json.load(f)

      config = {key: weight_path + "/" + str(value) for (key, value) in weights.items()}
      
      # config['num_features'] = weights['num_features']
      return config
      
    def get_conventional_densities(self, file): 
      preprocessor = Preprocessor(width=256, height=224, interpolation=3)
      dicom = preprocessor.load_dicom(file)
      manufacturer = DicomManager.get_manufacturer(dicom)
      normalization_config = self.GE_MD_NORMALIZER_CONFIG if manufacturer == 'GE' else self.HO_MD_NORMALIZER_CONFIG
      normalizer = CLAHENormalizer(**normalization_config)
      model = MultiDensity(preprocessor=preprocessor, normalizer=normalizer)
      model.model = self.GE_MD if manufacturer == 'GE' else self.HO_MD
      da = model.get_multilevel_da(file)
      ba = preprocessor.get_ba(file, width=256, height=224)
      da_cm2 = preprocessor.pixel_to_cm2(da, width=256, height=224, manufacturer=manufacturer)
      da_cm2.append(ba)
      return da_cm2
    
    def get_conventional_densities_all(self, files): 
      preprocessor = Preprocessor(width=256, height=224, interpolation=3)
      ge_model = MultiDensity(preprocessor=preprocessor, 
                              normalizer=CLAHENormalizer(**self.GE_MD_NORMALIZER_CONFIG))
      ge_model.model = self.GE_MD 
      ho_model = MultiDensity(preprocessor=preprocessor, 
                              normalizer=CLAHENormalizer(**self.HO_MD_NORMALIZER_CONFIG))
      ho_model.model = self.HO_MD
      
      result = np.zeros((len(files), 4))
      for i, file in enumerate(files): 
        print(f"{i+1}/{len(files)} Getting conventional densities from {file} ...")
        dicom = preprocessor.load_dicom(file)
        manufacturer = DicomManager.get_manufacturer(dicom)
        print(f"Manufacturer : {manufacturer}")
        model = ge_model if manufacturer == 'GE' else ho_model 
        da = model.get_multilevel_da(file)
        ba = preprocessor.get_ba(file, width=256, height=224)
        da = preprocessor.pixel_to_cm2(da, width=256, height=224, manufacturer=manufacturer)
        da.append(ba)
        result[i,:] = np.array(da)
      return result
    
    def get_density_score(self, file):
      preprocessor = Preprocessor(width=256, height=224, interpolation=3)
      dicom = preprocessor.load_dicom(file)
      manufacturer = DicomManager.get_manufacturer(dicom)
      normalization_config = self.GE_MD_NORMALIZER_CONFIG if manufacturer == 'GE' else self.HO_MD_NORMALIZER_CONFIG
      normalizer = CLAHENormalizer(**normalization_config)
      
      lr = LogisticRegression()
      coef = self.GE_DR_COEF if manufacturer == 'GE' else self.HO_DR_COEF
      intercept = self.GE_DR_INT if manufacturer == 'GE' else self.HO_DR_INT
      lr.coef_ = np.load(coef)
      lr.intercept_ = np.load(intercept)
      model = DenseRisk(preprocessor=preprocessor, normalizer=normalizer)
      model.risk_model = lr 
      model.num_feature = self.DR_NUM_FEATURES 
      model.scaler = self.GE_DR_SCALER if manufacturer == 'GE' else self.HO_DR_SCALER

      return model.get_density_score(file)
    
    def get_density_score_all(self, files): 
      result = [0 for file in files]
      for i, file in enumerate(files): 
          result[i] = self.get_density_score(file)
      return result
    
    def get_deep_mammo_risk(self, file): 
      preprocessor = Preprocessor(width=224, height=224, interpolation=3)
      dicom = preprocessor.load_dicom(file)
      manufacturer = DicomManager.get_manufacturer(dicom)
      normalization_config = self.GE_MR_NORMALIZER_CONFIG if manufacturer == 'GE' else self.HO_MR_NORMALIZER_CONFIG
      normalizer = CLAHENormalizer(**normalization_config)
      model = DeepMammoRisk(preprocessor=preprocessor, normalizer=normalizer)
      model.model = self.GE_MR if manufacturer == 'GE' else self.HO_MR

      return model.get_deep_mammo_score(file)
    
    def get_deep_mammo_risk_all(self, files): 
      preprocessor = Preprocessor(width=224, height=224, interpolation=3)
      ge_model = DeepMammoRisk(preprocessor=preprocessor, 
                               normalizer=CLAHENormalizer(**self.GE_MR_NORMALIZER_CONFIG))
      ge_model.model = self.GE_MR 
      ho_model = DeepMammoRisk(preprocessor=preprocessor, 
                                    normalizer=CLAHENormalizer(**self.HO_MR_NORMALIZER_CONFIG))
      ho_model.model = self.HO_MR
      
      result = [0 for file in files]
      for i, file in enumerate(files): 
        print(f"{i+1}/{len(files)} Getting deep mammo risk score from {file} ...")
        dicom = preprocessor.load_dicom(file)
        manufacturer = DicomManager.get_manufacturer(dicom)
        print(f"Manufacturer : {manufacturer}")
        model = ge_model if manufacturer == 'GE' else ho_model 
        result[i] =  model.get_deep_mammo_score(file)
      return result
  
    def mammo2risk(self, files): 
      print("Run mammo2risk-v0.1.0")    
      result = pd.DataFrame()
      
      # Path parsing
      print("Loading files...")
      folder_names = [os.path.split(x)[0] for x in files]
      file_names = [os.path.split(x)[1] for x in files]
      dicoms = [Preprocessor.load_dicom(x) for x in tqdm(files)]
      
      # Dicom info extraction
      print("Extracting dicom information...")
      views = [DicomManager.get_view_position(x) for x in tqdm(dicoms)]
      side = [DicomManager.get_laterality(x) for x in tqdm(dicoms)]
      
      # mammo2risk 
      print("Loading Conventional Density Model...")
      densities = self.get_conventional_densities_all(files)
      
      print("Getting Density Scores...")
      dense_risk_scores = self.get_density_score_all(files)
      
      print("Loading Mammo Risk Model...")
      mammo_risk_scores = self.get_deep_mammo_risk_all(files)

      # Save files
      result['folder'] = folder_names
      result['file'] = file_names
      result["view"] = views
      result["side"] = side
      result["cumulus(cm2)"] = densities[:, 0]
      result["alto_cumulus(cm2)"] = densities[:, 1]
      result["cirro_cumulus(cm2)"] = densities[:, 2]
      result["breast_area(cm2)"] = densities[:, 3]
      result["denserisk"] = dense_risk_scores
      result["mammorisk"] = mammo_risk_scores 
      return result
      
if __name__ == "__main__":
    pass