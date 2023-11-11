#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 08:58:04 2023

@author: maya
"""



import __init__
from tensorflow.keras.applications import MobileNet, ResNet50

from tensorflow.keras import optimizers, callbacks

#import keras_tuner

from Dataset_loaders.datasets import CalebA



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model_archetecture', choices = ['MobileNet', 'ResNet50'])
##ToDO: add dataset option and hyperparameter options.

args = parser.parse_args()

dset = CalebA(True, False)

train_dset, val_dset = dset.get_dset('train'), dset.get_dset('val')
from keras import Input, Model
from keras.layers import Dense
def load_model(args):

    if args.model_archetecture == 'ResNet50':
        m = ResNet50(classes = 40,weights = None,input_shape=[224,224,3], classifier_activation = 'sigmoid')
        w = ResNet50(classifier_activation = None).get_weights()
        w[-2] = w[-2][:,:40]
    if args.model_archetecture == 'MobileNet':
        m = MobileNet(classes = 40,weights = None,input_shape=[224,224,3], classifier_activation = 'sigmoid')
        w = MobileNet(classifier_activation = None).get_weights()
        w[-2] = w[-2][:,:,:,:40]

    w[-1] = w[-1][:40]
    m.set_weights(w) 
    return m

m = load_model(args)
m.compile(optimizer = optimizers.Adam(learning_rate = 0.006), loss =  'binary_crossentropy', metrics = 'accuracy')
m.fit(train_dset, validation_data = val_dset, callbacks = callbacks.ModelCheckpoint(args.model_archetecture + '50_170by170to224by224.h5',save_best_only=True, save_weights_only = True, monitor = 'val_accuracy', initial_value_threshold = .5), epochs = 100)




def build_model(hp):
    m = load_model(args)
    optim = optimizers.Adam #getattr(optimizers, hp.Choice('optimizer', ['Adam', 'AdamW'] ))
    loss = hp.Choice('loss_type', ['binary_crossentropy', 'binary_focal_crossentropy'])
    m.compile(optimizer = optim(lr = hp.Float('lr', min_value = 1e-4, max_value = 1e-2, sampling = 'log')), loss = loss, metrics = 'binary_accuracy')
    
    return m

#tuner = keras_tuner.Hyperband(build_model, 'val_binary_accuracy', project_name = 'ResNet50')
#tuner.search(train_dset, validation_data = val_dset)
