# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 06:48:52 2017

@author: Brilian
"""

import tensorflow as tf
from mnist import MNIST

sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, shape=[float,28x28]) #image 28x28 pixels
Y = tf.placeholder(tf.float32, shape=[float, 10]) #class 0...9

#initialize weight and bias

W = tf.Variable(tf.random_normal([784,10]), tf.float32)
b = tf.Variable(tf.random_normal([10]), tf.float32)

#initialize all variables
sess.run(tf.global_variables_initializer())

predict = tf.nn.softmax(tf.matmul(X, W) + b)
cost_func = tf.reduce_mean( -tf.reduce_mean(y * tf.log(y), reduction_indices=[1]) )

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost_func)

for i in range(1000):
    batch = 