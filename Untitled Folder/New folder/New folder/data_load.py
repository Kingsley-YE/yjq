import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras

def getALLImages(folder):
    assert os.path.exists(folder)
    assert os.path.isdir(folder)
    imageList = os.listdir(folder)
    imageList = [os.path.abspath(item) for item in imageList if os.path.isfile(os.path.join(folder,item))]
    
    return imageList
    
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def load_data(path):
    num_class = 40
    image = []
    label = []
    for nc in range(num_class):
        c = get_imlist(path+str(nc))
        d = len(c)
        for i in range(d):
            img = Image.open(c[i])
            image.append(np.asarray(img,dtype='float64'))
            label.append(nc)

    im = np.array(image)
    lab = np.array(label)   
    return im,lab

def get_shuffle_data(X, y, num = 14801):
    num_train = int (num * 0.9)
    num_test = num - num_train
    num_training = int( num_train * 0.9)
    num_val = num_train - num_training
    
    #y = y.reshape(num,1)
    #X =  np.reshape(X, (X.shape[0], -1))
    
    index = np.arange(num)
    np.random.shuffle(index)
    X_train = X[index]
    y_train = y[index]
    
  
    X_test = X_train[:num_test,:]
    X_train = X_train[num_test:,:]
    X_val = X_train[:num_val,:]
    index1=index[num_test:]
    index2=index1[:num_val]
    np.save("index.npy", index2)
    X_train = X_train[num_val:,:]
    
    y_test = y_train[:num_test]
    y_train = y_train[num_test:]
    y_val = y_train[:num_val]
    y_train = y_train[num_val:]
    
    # mask = np.random.choice(num_training, num_dev, replace=False)
    # X_dev = X_train[mask]
    # y_dev = y_train[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # Transpose so that channels come first
    #X_train = X_train.transpose(0, 3, 1, 2).copy()
    #X_val = X_val.transpose(0, 3, 1, 2).copy()
    #X_test = X_test.transpose(0, 3, 1, 2).copy()

    return X_train, y_train, X_val, y_val, X_test, y_test