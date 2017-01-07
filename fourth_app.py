# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 18:12:39 2017

@author: Brilian
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_act(i = 1.0, act_funct = lambda x: x):
    ws = np.arange(-0.5, 0.5, 0.05)
    bs = np.arange(-0.5, 0.5, 0.05)
    X, Y = np.meshgrid(ws, bs)
    os = np.array([act_funct(tf.constant(w*i + b)).eval(session=sess) \
                            for w,b in zip(np.ravel(X) , np.ravel(Y))])
    Z = os.reshape(X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1)

def activation_func(X): return X
sess = tf.Session() 
#with tf.Session() as sess:
inp = tf.constant([1.0, 2.0, 3.0], shape=[1,3])
ws = tf.random_normal([3,3])
bs = tf.random_normal([1,3])

#act = activation_func(tf.matmul(inp,ws) + bs)
#act = tf.sigmoid(tf.matmul(inp,ws) + bs) #sigmoid
#act = tf.tanh(tf.matmul(inp,ws) + bs) #tanh
act = tf.nn.relu(tf.matmul(inp,ws) + bs) #reLU
act.eval(session = sess)
plot_act(1.0, activation_func)
#plot_act(1.0, tf.sigmoid)
plot_act(1.0, tf.tanh)
