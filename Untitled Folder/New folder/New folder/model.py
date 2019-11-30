import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import Transform

def simple_model(x,y,is_training):
    x_flat=tf.reshape(x,[-1,8192])
    fc=tf.layers.dense(inputs=x_flat,units=1024,activation=tf.nn.relu)
    bn=tf.layers.batch_normalization(inputs=fc,training=is_training)
    fc2=tf.layers.dense(inputs=bn,units=512,activation=tf.nn.relu)
    bn2=tf.layers.batch_normalization(inputs=fc2,training=is_training)
    y_out=tf.layers.dense(inputs=bn2,units=40)
    return y_out

