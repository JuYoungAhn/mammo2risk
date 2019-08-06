import sys
from mammo2risk.preprocessing import Preprocessor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import keras 
import warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn

class CorrelationMonitoring(keras.callbacks.Callback):
    BACKGROUND_REGION = -1
    BREAST_REGION = 0
    DENSE_REGION = 1
    AC_DENSE_REGION = 2
    CC_DENSE_REGION = 3
    
    def __init__(self, manufacturer) : 
        self.manufacturer = manufacturer

    def on_train_begin(self, logs={}):
        self._cu_corr = []
        self._ac_corr = []
        self._cc_corr = []
  
    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_pred_prob = self.model.predict(X_val)

        y_true_class = Preprocessor.prob_map_to_class(y_val)
        y_pred_class = Preprocessor.prob_map_to_class(y_pred_prob)
        
        das_true = Preprocessor.get_multilevel_das(y_true_class)
        das_pred = Preprocessor.get_multilevel_das(y_pred_class)
        
        da_cu_pred = [x[0] for x in das_pred]
        da_ac_pred = [x[1] for x in das_pred]
        da_cc_pred = [x[2] for x in das_pred]
        
        da_cu_true = [x[0] for x in das_true]
        da_ac_true = [x[1] for x in das_true]
        da_cc_true = [x[2] for x in das_true]
          
        print("Cumulus: ", np.corrcoef(da_cu_pred, da_cu_true)[0][1])
        print("Altocumulus: ", np.corrcoef(da_ac_pred, da_ac_true)[0][1])
        print("Cirrocumulus: ", np.corrcoef(da_cc_pred, da_cc_true)[0][1])
          
        self._cu_corr.append(np.corrcoef(da_cu_pred, da_cu_true)[0][1])
        self._ac_corr.append(np.corrcoef(da_ac_pred, da_ac_true)[0][1])
        self._cc_corr.append(np.corrcoef(da_cc_pred, da_cc_true)[0][1])

        return 

    def get_data(self):
        return (self._cu_corr, self._ac_corr, self._cc_corr)

def history_plot(history) :
    # summarize history for accuracy
    plt.plot(history.history['mean_iou'])
    plt.plot(history.history['val_mean_iou'])
    plt.title('model accuracy')
    plt.ylabel('mean_iou')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
