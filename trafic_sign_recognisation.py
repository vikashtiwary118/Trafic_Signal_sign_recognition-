# -*- coding: utf-8 -*-
import sys
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
# from updated_intensity_10_class import *
import random


epochs = 20

train_data_path = '/home/manish/vikash/Trafic_signal/trafic_sign_recognisation/training'
validation_data_path = '/home/manish/vikash/Trafic_signal/trafic_sign_recognisation/validation'

# train_data_path = '/home/pranjal-artivatic/Desktop/under2/Training'
# validation_data_path = '/home/pranjal-artivatic/Desktop/under2/Validation'

"""
Parameters
"""
img_width, img_height = 224, 224
batch_size = 10
samples_per_epoch = 10
validation_steps = 5
nb_filters1 = 32
nb_filters2 = 64
nb_filters3=128
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 3
lr = 0.0004

def train_model_3():
    model = Sequential()
    model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same", input_shape=(img_width, img_height, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

    model.add(Convolution2D(nb_filters3, conv2_size, conv2_size, border_mode="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(classes_num, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')


    validation_generator = test_datagen.flow_from_directory(
        validation_data_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    """
    Tensorboard log
    """
    log_dir = './tf-log/'
    tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit_generator(
        train_generator,
        samples_per_epoch=samples_per_epoch,
        epochs=epochs,
        validation_data=validation_generator)



    target_dir = './models/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    model.save('./models/major_3.h5')
    model.save_weights('./models/major_3_1.h5')



def test_model_3(img_path):
#/home/manish/vikash/proj

    test_model = load_model('/home/manish/vikash/Trafic_signal/models/major_3.h5')
    img = load_img(img_path,False,target_size=(img_width,img_height))
    desired_dim=(32,64)
   # x = img_to_array(img)
    img = cv2.imread(img_path)
    x= cv2.resize(img, desired_dim, interpolation=cv2.INTER_LINEAR)
    x = np.expand_dims(np.array(x), axis=0)
    preds = test_model.predict_classes(x)
    prob = test_model.predict_proba(x)
    print(preds[0], prob,'in test_model_3')
    return preds



train_model_3()
#test_model_3('/home/manish/vikash/Trafic_signal/trafic_sign_recognisation/training/green')


