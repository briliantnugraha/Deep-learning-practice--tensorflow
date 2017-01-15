# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 08:05:29 2017

@author: Brilian
"""

import _pickle as pickle, gzip, numpy as np
import tensorflow as tf
from multiclass import one_hot

def run():
    # unload the dataset file
    f = gzip.open('E:\Deep learning/mnist.pkl.gzip', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    
    #get the train dataset and train label
    train_x, train_y = train_set
    train_x = np.array(train_x)
    
    #make the train label into one hot encoding class
    total_class = len(set(train_y))
    train_label = one_hot(train_y, total_class)
    
    #make placeholder for the dataset and the one hot classes (10 classes)
    X = tf.placeholder(tf.float32, shape=[None,28*28]) #image 28x28 pixels
    Y = tf.placeholder(tf.float32, shape=[None, total_class]) #class 0...9
    
    #initialize weight and bias
    W = tf.Variable(tf.zeros([28*28,total_class]), tf.float32)
    b = tf.Variable(tf.zeros([total_class]), tf.float32)
    
    sess = tf.InteractiveSession() #initialize the tensorflow Session
    sess.run(tf.global_variables_initializer()) #initialize all variables
    
    #both the cost function technique could be used
    predict = tf.nn.softmax(tf.matmul(X, W) + b) #this two technique a little slower
    cost_func = tf.reduce_mean( \
           -tf.reduce_sum(Y*tf.log(predict), reduction_indices=[1]) )
    #predict = tf.matmul(X, W) + b
#    cost_func = tf.reduce_mean( \
#       tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits= predict) )
    
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost_func)
    for i in range(0, 50000, 50): #mini batch Gradient Descent
        batch_train_data = train_x[i:i+50]
        batch_train_label = train_label[i:i+50]
        
        #both technique below could be used
    #    sess.run(train_step, feed_dict={X: batch_train_data, Y:batch_train_label})
        train_step.run(feed_dict={X: batch_train_data, Y:batch_train_label})
        
    correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print (sess.run(accuracy, feed_dict={X: train_x, Y: train_label}))
    sess.close()
    
if __name__ == "__main__":
    run()