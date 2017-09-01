# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%matplotlib inline
from __future__ import division,print_function

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
import inspect


os.chdir('/Users/isaaclavine/fastai/courses/deeplearning1/nbs')
import utils; reload(utils)
from utils import plots
import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import tf as ktf
import scipy


# Set the important constants
batch_size=32
num_classes = 10



# Import the VGG model
import vgg16; reload(vgg16)
from vgg16 import Vgg16

# Import the data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Resize the images
new_shape = (224,224, 3)
x_train_new = np.empty(shape=(x_train.shape[0],)+new_shape)
x_test_new = np.empty(shape=(x_test.shape[0],)+new_shape)


print(x_train_new.shape)
print(x_test_new.shape)

for i in xrange(x_train.shape[0]):
    x_train_new[i] =  scipy.misc.imresize(x_train[i], new_shape)

print(x_train_new.shape)

#for j in xrange(x_test.shape[0]):
#    x_test_new[j] =  scipy.misc.imresize(x_test[j], new_shape)
    
    
#print(x_test_new.shape)

x_train = np.rollaxis(x_train_new, 3, 1)
#x_test = np.rollaxis(x_test_new, 3, 1)


print(x_train.shape)
#print(x_test.shape)

# Convert class vectors to binary class matrices.
y_train = np_utils.to_categorical(y_train, num_classes)
#y_test = np_utils.to_categorical(y_test, num_classes)
print(y_train.shape)
#print(y_test.shape)

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    dim_ordering = "th")
    
print(x_train.shape)


batches = datagen.flow(x_train, y_train, batch_size=batch_size)
                        #save_to_dir = '/Users/isaaclavine/fastai/courses/deeplearning1/nbs/data/CIFAR/',
                       #save_prefix = 'stuff',
                     # save_format = 'jpeg')

#val_batches = datagen.flow(x_test, y_test, batch_size = batch_size)
print(x_train.shape)

# Adding in some random constants
batches.nb_class = num_classes
batches.class_indices = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
batches.nb_sample = x_train.shape[0]

#val_batches.nb_class = num_classes
#val_batches.class_indices = {'zero':0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
#val_batches.nb_sample = x_test.shape[0]

# print(batches.class_indices)

# Define the vgg model
vgg = Vgg16()
# Grab a few images at a time for training and validation.
# NB: They must be in subdirectories named based on their category
vgg.finetune(batches)
vgg.classes
print(y_train.shape)

## Fitting the model
# vgg.model.fit(x = x_train, y = y_train, batch_size = 32, nb_epoch = 1, validation_data = (x_test, y_test))
vgg.model.fit(x = x_train, y = y_train, batch_size = 32, nb_epoch = 1)
#vgg.fit(batches, val_batches, nb_epoch=1)
#vgg.fit(batches, nb_epoch=1)
