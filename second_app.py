# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 15:18:44 2017

@author: Brilian
"""

import tensorflow as tf

state = tf.Variable(0) #this is used to handle variables
#a = tf.constant([[1,2,3], [2,5,7]])

one = tf.constant(1)
new_val = tf.add(state,one)
update = tf.assign(state,new_val)

init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    print (session.run(state))
    
    for _ in range(3):
        session.run(update)
        print (session.run(state))

#==============================================================================
#     placeholder is the place to compute the data outside tensorflow model
#==============================================================================

a = tf.placeholder(tf.float32)
b = a*2
dictionary = {a: [[1,2,5], [8,10,20]]}

with tf.Session() as sess:
    result = sess.run(b, feed_dict=dictionary)#{a: 4.7})
    print (result)