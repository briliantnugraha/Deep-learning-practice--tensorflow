# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#this is a simple adding in tensorflow

import tensorflow as tf

a = tf.constant([2])
b = tf.constant([3])

c = tf.add(a,b)


with tf.Session() as session:
    result = session.run(c)
    print (result)