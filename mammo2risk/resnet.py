from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential

def ResNet(top_fix=False, do=0.5, weights='imagenet'): 
    model = Sequential()
    resnet = ResNet50(include_top=False, input_shape=(224, 224, 3), weights=weights)

    if top_fix: 
      for layer in resnet.layers:
          layer.trainable = False

    model.add(resnet)
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(do))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(do))
    model.add(Dense(1, activation="sigmoid"))
    return model
  
def ResNetWOsigmoid(top_fix=False, do=0.5, weights='imagenet'): 
    model = Sequential()
    resnet = ResNet50(include_top=False, input_shape=(224, 224, 3), weights=weights)

    if top_fix: 
      for layer in resnet.layers:
          layer.trainable = False

    model.add(resnet)
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(do))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(do))
    model.add(Dense(1, activation=None))
    return model