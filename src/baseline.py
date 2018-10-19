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
DEBUG = False

#loading our dataset
(x_train, y_train), (x_validation, y_validation) = fashion_mnist.load_data()
y_validation = keras.utils.to_categorical(y_validation, 10)
y_train = keras.utils.to_categorical(y_train, 10)

# adding a color channel to those grayscale images
x_validation = numpy.expand_dims(x_validation, axis=3)
x_train = numpy.expand_dims(x_train, axis=3)

#normalisation
x_validation = x_validation * 1./255
x_train = x_train * 1./255

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
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    brightness_range=None, 
    shear_range=0.2, 
    zoom_range=0.2, 
    fill_mode='nearest', 
    horizontal_flip=False, 
    vertical_flip=False, 
    rescale=None, 
    preprocessing_function=None)

valid_gen = ImageDataGenerator()

#creating callbacks:
callback_list = []
callback_list.append(callbacks.EarlyStopping(patience=7))
callback_list.append(callbacks.ReduceLROnPlateau(patience=5))

path = None
#creating our result directory
if(DEBUG == False):
    path = "result/test_baseline_" + str(time.time())
    os.makedirs(path, exist_ok=True)
    path += "/"
    #callback_list.append()

lenet = model.LeNet5(train_gen, valid_gen, step_epoch, batch_size)
lenet.build_model(input_shape, optimizer, loss_funct, number_classes)

hist_dict = lenet._step_train(x_train=x_train, y_train=y_train, x_validation=x_validation, y_validation=y_validation)

if(DEBUG == False):
    val_loss = hist_dict['val_loss']
    val_accuracy = hist_dict['val_acc']
    loss = hist_dict['loss']
    accuracy = hist_dict['acc']

    plt.plot(val_loss, label="validation loss")
    plt.plot(loss, label="train loss")
    plt.title('Loss evolution')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.savefig(path + "loss.png")

    plt.plot(val_loss, label="validation accuracy")
    plt.plot(loss, label="train accuracy")
    plt.title('Accuracy evolution')
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.savefig(path + "accuracy.png")