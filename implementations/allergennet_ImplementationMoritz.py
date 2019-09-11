#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Activation, Dense, Flatten, Reshape, Conv1D, Conv2D, Add, BatchNormalization, MaxPooling1D)


# In[2]:


def res_layer(filter, input): #pseudo residual layer zuerst ohne und dann mit pooling
    aconv1d = Conv1D(filters=filter, kernel_size=3, padding='same', activation='relu')(input)
    #padding=same --> filter verwirft bei der Berechnung den ersten Eintrag nicht, weil zur berechnung ein "-1te" Wert als imaginärer Wert angenommen wird
    bconv1d = Conv1D(filters=filter, kernel_size=3, padding='same', activation='relu')(aconv1d)
    cconv1d = Conv1D(filters=filter, kernel_size=1, padding='same', activation='relu')(bconv1d)
    normalized = BatchNormalization()(cconv1d)
    raconv1d = Conv1D(filters=filter, kernel_size=3, padding='same', activation='relu')(normalized)
    lconv1d = Conv1D(filters=filter, kernel_size=3, padding='same', activation='relu')(normalized)
    rbconv1d = Conv1D(filters=filter, kernel_size=3, padding='same', activation='relu')(raconv1d)
    rcconv1d = Conv1D(filters=filter, kernel_size=1, padding='same', activation='relu')(rbconv1d)
    rmaxpooling = MaxPooling1D(pool_size=2)(rcconv1d)
    lmaxpooling = MaxPooling1D(pool_size=2)(lconv1d)
    add = Add()([rmaxpooling, lmaxpooling])
    return add

def res_layer_wopooling(filter, input): #pseudo residual layer komplett ohne pooling
    aconv1d = Conv1D(filters=filter, kernel_size=3, padding='same', activation='relu')(input)
    bconv1d = Conv1D(filters=filter, kernel_size=3, padding='same', activation='relu')(aconv1d)
    cconv1d = Conv1D(filters=filter, kernel_size=1, padding='same', activation='relu')(bconv1d)
    normalized = BatchNormalization()(cconv1d)
    return normalized


def pooling_layer(filter, input):
    raconv1d = Conv1D(filters=filter, kernel_size=3, padding='same', activation='relu')(input)
    lconv1d = Conv1D(filters=filter, kernel_size=3, padding='same', activation='relu')(input)
    rbconv1d = Conv1D(filters=filter, kernel_size=3, padding='same', activation='relu')(raconv1d)
    rcconv1d = Conv1D(filters=filter, kernel_size=1, padding='same', activation='relu')(rbconv1d)
    rmaxpooling = MaxPooling1D(pool_size=2)(rcconv1d)
    lmaxpooling = MaxPooling1D(pool_size=2)(lconv1d)
    add = Add()([rmaxpooling, lmaxpooling])
    return add


# In[3]:



#embedding
myInput = Input(shape=(20,1912))
reshaped = Reshape(target_shape=(20,1912,1))(myInput)
myConv2d = Conv2D(filters=256, kernel_size=(20,1))(reshaped)
reshaped = Reshape(target_shape=(1912,256))(myConv2d)
normalized = BatchNormalization()(reshaped)


model = Model(myInput, normalized)
model.summary()


# In[4]:


temp=res_layer(256, normalized)
temp=res_layer(128, temp)
temp=res_layer(64, temp)
temp=res_layer(32, temp)
temp=res_layer(16, temp)
temp=res_layer(8, temp)
temp=res_layer(4, temp)
temp=res_layer_wopooling(2, temp)
temp=res_layer_wopooling(1, temp)
temp=pooling_layer(1, temp)
temp=Flatten()(temp)
temp=Dense(1, activation='softmax')(temp)#oder activation='sigmoid'
model = Model(myInput, temp)
model.summary()


# In[ ]:




