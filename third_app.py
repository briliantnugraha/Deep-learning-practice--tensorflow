# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 17:34:43 2017

@author: Brilian
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mp

#manual simple linear regression
#X = np.arange(0.0, 5.0, 0.1)
#a = 1
#b = 0
#Y = a*X + b
#
#plt.plot(X, Y)
#plt.ylabel("Dependent Variable")
#plt.xlabel("Independent Variable")
#plt.show()

#linear regression tensorflow
x_data = np.random.rand(100).astype(np.float32)

#for example, here is the desire output , or y_data, with a = 3, and b = 2
y_data = 3*x_data + 2
y_data = np.vectorize(lambda y: y + np.random.normal(scale=0.1))(y_data)
y_data = y_data.astype(np.float32)

#predicet the a and busing tensorflow
a = tf.Variable(1.0)
b = tf.Variable(0.1)
y = a*x_data + b

loss_func = tf.reduce_mean(tf.square(y - y_data))

#use simple gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.3)
train = optimizer.minimize(loss_func)

init = tf.global_variables_initializer() #new version of initialize_all_variables
train_data = []
with tf.Session() as sess:
    sess.run(init)
    for step in range(len(y_data)):
        evaluate = sess.run([train, a, b])[1:]
        if step % 5 == 0:
            print (step, evaluate)
            train_data.append(evaluate)
a, b = train_data[-1]
f_y = a*x_data + b

line = plt.plot(x_data, f_y)
plt.setp(line, color='g')
plt.plot(x_data, y_data, 'ro')
plt.ylabel("Dependent Variable")
plt.xlabel("Independent Variable")
plt.show()