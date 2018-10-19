import os
import time

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="3"


import keras

from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks as callbacks
import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model

#ENV
DEBUG = True

#loading our dataset
(x_val1, y_val1), (x_validation, y_validation) = fashion_mnist.load_data()
#concatenating (we only need one big validation dataset)

y_validation = keras.utils.to_categorical(y_validation, 10)


# adding a color channel to those grayscale images
x_validation = numpy.expand_dims(x_validation, axis=3)

#normalisation
x_validation = x_validation * 1./255

# labelisation of the 10 first images:
x_train = []
y_train = []
for position in range(10):
    #shift the image and the label to the train set
    x_train.append(x_validation[position])
    y_train.append(y_validation[position])
    #removing this element from the validation set
    x_validation = numpy.delete(x_validation, position, axis=0)
    y_validation = numpy.delete(y_validation, position, axis=0)

x_train = numpy.array(x_train)
y_train = numpy.array(y_train)


# model parameters
loss_funct = 'categorical_crossentropy'
optimizer = 'adam'
size_x = x_train.shape[1]
size_y = x_train.shape[2]
input_shape = (size_x, size_y, 1) #grayscale
number_classes = 10
step_epoch = 20
batch_size = 128

#data generator
train_gen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.0, 
    height_shift_range=0.0, 
    brightness_range=None, 
    shear_range=0.0, 
    zoom_range=0.0, 
    fill_mode='nearest', 
    horizontal_flip=False, 
    vertical_flip=False, 
    rescale=None, 
    preprocessing_function=None)

valid_gen = ImageDataGenerator()

#creating callbacks:
callback_list = []

path = None
#creating our result directory
if(DEBUG == False):
    path = "result/test_" + str(time.time())
    os.makedirs(path, exist_ok=True)
    path += "/"
    callback_list.append()

lenet = model.LeNet5(train_gen, valid_gen, step_epoch, batch_size)
lenet.build_model(input_shape, optimizer, loss_funct, number_classes)
smth = lenet.train(x_train, y_train, x_validation, y_validation)

import ipdb; ipdb.set_trace()