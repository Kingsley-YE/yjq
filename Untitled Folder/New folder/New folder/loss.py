import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

def loss(y_out,y):
    mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,40), logits=y_out))
    #define our optimizer
    optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
    train_step = optimizer.minimize(mean_loss)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(mean_loss)
    return mean_loss,train_step