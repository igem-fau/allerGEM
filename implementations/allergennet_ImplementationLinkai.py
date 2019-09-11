#!/usr/bin/env python
# coding: utf-8

# # ResNet(30)

# In[ ]:


import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Activation, Dense, Flatten, Reshape, Conv1D, Conv2D, Add, BatchNormalization, MaxPooling1D)


# ## Function and Co. definitions

# ### Map AA Codes to intergeters for one hot encoding

# In[2]:


aa1 = list("ACDEFGHIKLMNPQRSTVWY")
aa2int = dict((c, i) for i, c in enumerate(aa1))


# ### Function to one hot encode a AA code string

# In[ ]:


def onehot(seq):
    """Return the amino acid sequence as a one hot coded numpy array"""
    ret = np.zeros([len(1912), 20]) # the number 1912 may change

    for i, j in enumerate(seq):
        ret[i][aa2int[j]] = 1

    return ret


# ### Funtion to create a ResNet layer

# In[ ]:


def resblock(input_, filters_):
    """Creates a Residual Layer with x amount of filters. Returns the bottom
    layer.
    """
    lconv1d = Conv1D(filters=filters_, kernel_size=3)(input_)
    lmaxpooling = MaxPooling1D(pool_size=2)(lconv1d)
    convshape = tf.keras.backend.int_shape(lmaxpooling)
    
    raconv1d = Conv1D(filters=filters_, kernel_size=3)(input_)
    rbconv1d = Conv1D(filters=filters_, kernel_size=3)(raconv1d)
    rcconv1d = Conv1D(filters=filters_, kernel_size=1)(rbconv1d)
    rmaxpooling = MaxPooling1D(pool_size=2)(rcconv1d)
    reshaped = Reshape(target_shape=convshape[1:])(rmaxpooling)
    add = Add()([lmaxpooling, reshaped])

    return add


# ## Read input file and parse it 

# In[ ]:


data = list(open("../data/filtered.fasta", 'r'))
data = [i[:-1] for i in data]
labels = np.array([i.split(';')[1] for i in data])
data = [i.split(';')[0] for i in data]

# TODO: Use onhot function and create "training data format".


# ## ResNet Implementation

# In[ ]:


n_filters = 256
inputlayer = Input(shape=(20,1912000)) # This number must be ridiculously high
reshaped = Reshape(target_shape=(20,1912000,1))(inputlayer)
conv2d = Conv2D(filters=n_filters, kernel_size=(20,1))(reshaped)
reshaped = Reshape(target_shape=(1912000,n_filters))(conv2d)
normalized = BatchNormalization()(reshaped)
rb = resblock(normalized, n_filters)

for i in range(7):
    rb = resblock(rb, n_filters)
    # May need to cut down the number of layers so we don't need that huge number at the beginning
    rb = resblock(rb, n_filters)
    n_filters = int(n_filters/2)

rb = resblock(rb, n_filters)
n_filters = int(n_filters/2)
rb = resblock(rb, n_filters)

model = Model(inputlayer, rb)
print(model.summary())

