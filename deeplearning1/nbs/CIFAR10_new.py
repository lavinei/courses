#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 18:28:58 2017

@author: isaaclavine
"""

from __future__ import division,print_function

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
import inspect

import utils; reload(utils)
from utils import plots
import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
#from keras.backend import tf as ktf
import scipy
import tensorflow as tf

batch_size=32
import vgg16; reload(vgg16)
from vgg16 import Vgg16

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
sample = True
sample_size = 100


if sample:
    x_train = x_train[:sample_size]
    x_test = x_test[:sample_size]
    y_train = y_train[:sample_size]
    y_test = y_test[:sample_size]
    
num_classes = 10

x_train = np.rollaxis(x_train, 1, 4)
x_test = np.rollaxis(x_test, 1, 4)

sess = tf.InteractiveSession()

# Let's try to resize the images using tensor flow instead of scipy
new_shape = (224, 224)
x_train_new = np.empty(shape=(x_train.shape[0],)+new_shape)
x_test_new = np.empty(shape=(x_test.shape[0],)+new_shape)


x_train = tf.image.resize_images(x_train, new_shape).eval()
x_test = tf.image.resize_images(x_test, new_shape).eval()

sess.close()

# Convert the channel axis back to the place expected by our preset program
x_train = np.rollaxis(x_train, 3, 1)
x_test = np.rollaxis(x_test, 3, 1)

# Convert class vectors to binary class matrices.
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# Using Keras's module for generating batches, instead of going straight into the fitting process
# Either way works though, it's not a big deal! This is more useful to know how to do
# Because I think it can automatically standardize images, and also generate new fake images
# That are noisy transformations of the original
datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    dim_ordering = "th")

batches = datagen.flow(x_train, y_train, batch_size=batch_size)
val_batches = datagen.flow(x_test, y_test, batch_size = batch_size)

# Setting some constants that are generated automatically when keras reads in data from a directory
# But not when it's read in online as it is now
batches.nb_class = num_classes
batches.class_indices = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
batches.nb_sample = x_train.shape[0]

val_batches.nb_class = num_classes
val_batches.class_indices = {'zero':0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
val_batches.nb_sample = x_test.shape[0]

# Now defining the VGG model and telling it that we just want to finetune the existing model
vgg = Vgg16()
vgg.finetune(batches)

# Finally, fitting the model
vgg.fit(batches, val_batches, nb_epoch=1)

