# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 08:05:29 2017

@author: Brilian
"""

import _pickle as pickle, gzip, numpy as np
import tensorflow as tf
from multiclass import one_hot
from sklearn.utils import shuffle

def set_one_hot(train_y, valid_y, test_y):
    total_class = len(set(train_y))
    train_label = one_hot(train_y, total_class)
    valid_label = one_hot(valid_y, total_class)
    test_label = one_hot(test_y, total_class)
    return total_class, train_label, valid_label, test_label    
    
def unload_mnist(filepath = 'E:/Deep learning/mnist dataset/mnist.pkl.gzip'):
    # unload the dataset file
    f = gzip.open(filepath, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    return train_set, valid_set, test_set 
    
def run():
    #unload gzip file that contains mnist dataset
    train_set, valid_set, test_set  = unload_mnist()

    #get the train,valid, and testing dataset and labels
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x, test_y = test_set
    train_x, valid_x, test_x = np.array(train_x), np.array(valid_x), np.array(test_x)

    #make the train label into one hot encoding class
    total_class, train_label, valid_label, test_label = set_one_hot(train_y, valid_y, test_y)
    #this is essential to shuffle train data and label for batch/stochastic gradient descent    
    train_x, train_label = shuffle(train_x, train_label) 
        
    #make placeholder for the dataset and the one hot classes (10 classes)
    X = tf.placeholder(tf.float32, shape=[None,28*28]) #image 28x28 pixels
    Y = tf.placeholder(tf.float32, shape=[None, total_class]) #class 0...9    

    W = tf.Variable(tf.zeros([28*28,total_class]), tf.float32)#initialize weight and bias
    b = tf.Variable(tf.zeros([total_class]), tf.float32) #initialize weight and bias
    keep_prob = tf.placeholder(tf.float32)
    
    #both the cost function technique could be used
    predict = tf.nn.softmax(tf.matmul(X, W) + b) #this two technique a little slower
    cost_func = tf.reduce_mean( \
           -tf.reduce_sum(Y*tf.log(predict), reduction_indices=[1]) )
    #predict = tf.matmul(X, W) + b
#    cost_func = tf.reduce_mean( \
#       tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits= predict) )
    
#    train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cost_func) #standard SGD 
    train_step = tf.train.MomentumOptimizer(0.25, 0.7).minimize(cost_func) #use momentum
#    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost_func)
#    train_step = tf.train.RMSPropOptimizer(learning_rate=0.3).minimize(cost_func) #use RMS optimizer from Geoffrey Hinton

    
    sess = tf.InteractiveSession() #initialize the tensorflow Session
    sess.run(tf.global_variables_initializer()) #initialize all variables    
    
    batch = 100
    for i in range(0, 50000, batch): #mini batch Gradient Descent
        batch_train_data, batch_train_label = train_x[i:i+batch], train_label[i:i+batch]
    #    sess.run(train_step, feed_dict={X: batch_train_data, Y:batch_train_label})
        train_step.run(feed_dict={X: batch_train_data, Y:batch_train_label, keep_prob: 1.0}) #both technique below could be used
        
    correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy training\t: ", sess.run(accuracy, feed_dict={X: train_x, Y: train_label, keep_prob: 1.0}))
    print ("Accuracy validation\t: ", sess.run(accuracy, feed_dict={X: valid_x, Y: valid_label, keep_prob: 1.0}))
    print ("Accuracy testing\t: ", sess.run(accuracy, feed_dict={X: test_x, Y: test_label, keep_prob: 1.0}))
    sess.close()
    
if __name__ == "__main__":
    run()