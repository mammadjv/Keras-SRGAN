#!/usr/bin/env python
#title           :Utils_model.py
#description     :Have functions to get optimizer and loss
#author          :Deepak Birla
#date            :2018/10/30
#usage           :imported in other files
#python_version  :3.5.4

from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
import keras
from keras.layers import Concatenate


class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes MSE loss
    def MSE_loss(self, y_true, y_pred):
        generated_images_sr = Concatenate(axis=3)([y_pred, y_pred, y_pred])
        return K.mean(K.square(y_true - generated_images_sr))
    
def get_optimizer():
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#        adam = Adam(lr=1E-4)
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #    initial_learning_rate=1e-2,
    #    decay_steps=10000,
    #    decay_rate=0.9,
    #)
#    adam = keras.optimizers.SGD(learning_rate=0.01)
    return adam


