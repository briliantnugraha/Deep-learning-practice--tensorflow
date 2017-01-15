# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 08:05:29 2017

@author: Brilian
"""

import _pickle as pickle, gzip, numpy as np
import tensorflow as tf

# Load the dataset
f = gzip.open('E:\Deep learning/mnist.pkl.gzip', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set
train_x = np.array(train_x)
train_y = np.array(train_y)
#print (train_x.shape, train_y.shape[0])

train_label = np.zeros([train_y.shape[0], 10])
for i in range(train_y.shape[0]):
    train_label[i,train_y[i]] = 1

sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, shape=[None,28*28]) #image 28x28 pixels
Y = tf.placeholder(tf.float32, shape=[None, 10]) #class 0...9

#initialize weight and bias

W = tf.Variable(tf.zeros([784,10]), tf.float32)
b = tf.Variable(tf.zeros([10]), tf.float32)

#initialize all variables
sess.run(tf.global_variables_initializer())

predict = tf.nn.softmax(tf.matmul(X, W) + b)
cost_func = tf.reduce_mean( -tf.reduce_mean(predict * tf.log(predict)) )
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost_func)
batch = 0
for i in range(1000):
    batch_train_data = train_x[batch:batch+50]
    batch_train_label = train_label[batch:batch+50]
    train_step.run(feed_dict={X: batch_train_data, Y:batch_train_label})
    batch += 50
    
correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print (train_label[0])
#acc = accuracy.eval()

sess.close()