#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 08:58:04 2023

@author: maya
"""


        


from tensorflow.keras.applications import MobileNet, ResNet50

from tensorflow.keras import optimizers, callbacks

#import keras_tuner

from Dataset_loaders.datasets import CalebA



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model_archetecture', choices = ['MobileNet', 'ResNet50'])


args = parser.parse_args()

dset = CalebA(True, False)

train_dset, val_dset = dset.get_dset('train'), dset.get_dset('val')

def load_model(args):
    m = ResNet50(classes = 40,weights = None,input_shape=[218,178,3])
    w = ResNet50().get_weights()

    w[-1] = w[-1][:40]
    w[-2] = w[-2][:,:40]
    m.set_weights(w) 
    return m

m = load_model(args)
m.compile(optimizer = optimizers.Adam(learning_rate = 0.004), loss =  'binary_crossentropy', metrics = 'binary_accuracy')
m.fit(train_dset, validation_data = val_dset, callbacks = callbacks.ModelCheckpoint('ResNet50.h5',save_best_only=True, save_weigts_only = True, monitor = 'val_binary_accuracy', initial_value_threshold = .8), epochs = 100)



def build_model(hp):
    m = load_model(args)
    optim = getattr(optimizers, hp.Choice('optimizer', ['Adam', 'AdamW'] ))
    loss = hp.Choice('loss_type', ['binary_crossentropy', 'binary_focal_crossentropy'])
    m.compile(optimizer = optim(lr = hp.Float('lr', min_value = 1e-4, max_value = 1e-2, sampling = 'log')), loss = loss, metrics = 'binary_accuracy')
    
    return m

#tuner = keras_tuner.Hyperband(build_model, 'val_binary_accuracy', factor = 2, project_name = 'ResNet50')
#tuner.search(train_dset, validation_data = val_dset, epochs = 20)
