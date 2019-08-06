import numpy as np
from tqdm import tqdm
import pandas as pd 
import sys
import warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn

class DicomManager(object) : 
    @classmethod  
    def get_manufacturer(cls, dicom) : 
        try :
            if 'ge' in dicom.Manufacturer.lower() : 
                result = "GE"
            elif 'holo' or 'lo' in dicom.Manufacturer.lower() : 
                result = "HOLOGIC"
            else : 
                result = dicom.Manufacturer
        except : 
            result = "Missing"
        return result

    @classmethod
    def get_model(cls, dicom) : 
        try :
            result = dicom.ManufacturerModelName
        except : 
            result = "Missing"
        return result
    
    @classmethod
    def get_patient_name(cls, dicom) :
        try : 
            name = dicom.PatientName
        except : 
            name = "Missing"
        return name

    @classmethod  
    def get_protocol(cls, dicom) : 
        result = "Missing"
        try : 
            protocol = dicom.ProtocolName

            if "combo" in protocol.lower() :
                result = "Combo"

            elif "tomo" in protocol.lower() : 
                result = "Tomo"

            else : 
                result = protocol
        except : 
            protocol = "Missing"

        return result

    @classmethod  
    def get_view_position(cls, dicom) : 
        try :
            result = dicom.ViewPosition
        except : 
            result = "Missing"
        return result
    
    @classmethod
    def get_laterality(cls, dicom) : 
        try :
            result = dicom.ImageLaterality
        except : 
            result = "Missing"
        return result
    
    @classmethod
    def get_institution(cls, dicom) : 
        try :
            result = dicom.InstitutionName
        except : 
            result = "Missing"
        return result
    
    @classmethod
    def get_exposure(cls, dicom) : 
        try :
            result = dicom.Exposure
        except : 
            result = "Missing"
        return result
    
    @classmethod
    def get_series_description(cls, dicom) : 
        try :
            result = dicom.SeriesDescription
        except : 
            result = "Missing"
        return result
    
    @classmethod
    def get_study_description(cls, dicom) : 
        try :
            result = dicom.StudyDescription
        except : 
            result = "Missing"
        return result
    
    @classmethod
    def get_exposure_time(cls, dicom) : 
        try :
            result = dicom.ExposureTime
        except : 
            result = "Missing"
        return result
    
    @classmethod
    def get_view_position(cls, dicom) : 
        try :
            result = dicom.ViewPosition
        except : 
            result = "Missing"
        return result
    
    @classmethod
    def print_dicom_information(cls, dicom) : 
        print("Exposure : ", cls.get_exposure(dicom))
        print("Exposure Time : ", cls.get_exposure_time(dicom))
        print("Institute : ", cls.get_institution(dicom))
        print("Manufacturor : ", cls.get_manufacturer(dicom))
        print("PatientName : ", cls.get_patient_name(dicom))
        print("Manufacturor Model Name : ", cls.get_model(dicom))
        print("View : ", cls.get_view_position(dicom))
        print("Protocol Name : ", cls.get_protocol(dicom))
        print("StudyDescription : ", cls.get_study_description(dicom))
    
    @classmethod
    def get_dicom_information(cls, dicom) : 
        result = {'exposure' : cls.get_exposure(dicom), 'exposure_time' : cls.get_exposure_time(dicom), 
                              'institute' : cls.get_institution(dicom), 'manufacturer' : cls.get_manufacturer(dicom), 
                              'patient' : cls.get_patient_name(dicom), 'model' : cls.get_model(dicom), 
                              'view' : cls.get_view_position(dicom), 'protocol' : cls.get_protocol(dicom), 'study_description' : cls.get_study_description(dicom)}
        return result
    
    @classmethod
    def get_2D_image_flag(cls, files) : 
        flag = []
        for file in tqdm(files):
            dicom = preprocessing.load_dicom(file)
            try : 
                image = preprocessing.get_dicom_pixel(dicom)
                if np.max(image) > 3000:
                       flag.append(True)
                else:
                     flag.append(False)
            except : 
                print(file + " has problem.")
                flag.append(False)
            return flag