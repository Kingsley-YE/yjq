import numpy as np
from tensorflow.keras.applications import VGG16
import data_load as du

def transform(path):
    x,y=du.load_data(path)
    X_train, y_train, X_val, y_val, X_test, y_test=du.get_shuffle_data(x, y)
    X_train/=255
    X_val/=255
    X_test/=255
    basemodel = VGG16(input_shape=(128,128,3),weights='imagenet', include_top=False)
    x_train=basemodel.predict(X_train)
    x_val=basemodel.predict(X_val)
    x_test=basemodel.predict(X_test)
    return x_train,x_val,x_test,y_train,y_val,y_test
