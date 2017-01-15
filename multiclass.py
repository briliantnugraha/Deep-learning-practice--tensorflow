# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 20:07:23 2017

@author: Brilian
"""
import numpy as np

def one_hot(Y, lenY):
    train_y = np.array(Y)
    train_label = np.zeros([train_y.shape[0], lenY])
    for i in range(train_y.shape[0]):
        train_label[i,train_y[i]] = 1
    return train_label
    