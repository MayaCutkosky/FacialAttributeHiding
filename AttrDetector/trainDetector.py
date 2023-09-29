#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 08:58:04 2023

@author: maya
"""
import numpy as np
from tensorflow.data import Dataset
from tensorflow.io import read_file, decode_jpeg
from tensorflow import cast, float32
from tensorflow.image import resize

class CalebA():
    def __init__(self, attributes, identities):
        self.identities = identities
        self.attr = attributes
        self.input_shape = [224,224, 3]
        self.dir = '/home/maya/Desktop/datasets/Celeba/'


    def get_dset(self, partition = 'train'):
        
        if partition == 'train':
            im_files = np.arange(1,162079)
            start = 0
            end = 40521
        elif partition == 'val':
            im_files = np.arange(162079, 182339)
            start = 162078
            end = 20261
        elif partition == 'test':
            im_files = np.arange(182339, 202599)
            start = 182338
            end = 0
        #randomly shuffle dataset
        random_shuffle = np.random.permutation(len(im_files))
        

        #image file names
        im_files = np.char.zfill(im_files.astype('S6'),6)[random_shuffle]
        # im_shapes = np.load(self.dir + 'im_sizes.npy')
        
        
        

        #load attr
        # self.attr = False
        if self.attr: #0 false, 1 true
            attr = (np.genfromtxt(self.dir + 'list_attr_celeba.txt',skip_header=2+start, skip_footer = end, usecols=np.arange(1,41)) + 1) // 2
            attr = attr[random_shuffle]
        
        #load image
        def load_image(im):
            f = read_file(self.dir + 'img_align_celeba/'  + im + '.jpg')
            im = decode_jpeg(f)
            im = cast(im, float32)/256
            return im
        
        
        def center_and_resize_image(im):
            # dim_x, dim_y = data[1]
            # crop_size = ((dim_x - dim_y))//2
            im = im[20:198]
            # im = resize(im, self.input_shape[:2])
            return im
        
        #create dataset
        x_dset = Dataset.from_tensor_slices(im_files)
        x_dset = x_dset.map(load_image)
        
        #x_dset = x_dset.map(center_and_resize_image)
        
        y_dset = Dataset.from_tensor_slices(attr)
        
        dset = Dataset.zip((x_dset,y_dset))
        dset = dset.batch(32)
        dset = dset.prefetch(4)
        
        return dset 

attr_names = ['5 o Clock Shadow', 'Arched Eyebrows', 'Attractive',
           'Bags Under Eyes', 'Bald', 'Bangs', 'Big Lips', 'Big Nose',
           'Black Hair', 'Blond Hair', 'Blurry', 'Brown Hair',
           'Bushy Eyebrows', 'Chubby', 'Double Chin', 'Eyeglasses',
           'Goatee', 'Gray Hair', 'Heavy Makeup', 'High Cheekbones',
           'Male', 'Mouth Slightly Open', 'Mustache', 'Narrow Eyes',
           'No Beard', 'Oval Face', 'Pale Skin', 'Pointy Nose',
           'Receding Hairline', 'Rosy Cheeks', 'Sideburns', 'Smiling',
           'Straight Hair', 'Wavy Hair', 'Wearing Earrings',
           'Wearing Hat', 'Wearing Lipstick', 'Wearing Necklace',
           'Wearing Necktie', 'Young']

from tensorflow.keras.losses import Loss
import tensorflow as tf
class BinaryCrossentropy(Loss):
    
    def __init__(self, **kwargs):
        self.alpha = 0.1
        self.loss_weight = self.add_weight(name = 'loss_weight', initializer = 'zeros')
        super().__init__(**kwargs)
        
    def call(self, y_true, y_pred):
        '''
        calculates binary crossentropy loss from our somewhat strange data format

        Parameters
        ----------
        y_true : batch_size by num_attr, float32 tensor, values = -1,0,1
            1 = True, 0 = unknown, -1 = False
        y_pred : batch_size by num_attr, float32 tensor, values in R
            <0.5 -> False, >0.5 -> True

        Returns
        -------
        loss : binary crossentropy loss

        '''
        
        #bad_vals = (y_true == 0) 
        
        loss_vector =  tf.clip_by_value(y_pred, 0, 1000) - y_pred * y_true + tf.math.log1p(1+ tf.exp(-tf.abs(y_pred)))
        loss_vector = tf.reduce_mean( loss_vector, axis = 0)
        
        
        self.loss_weight += tf.stop_gradient(self.loss_weight * (1-self.alpha) + self.alpha / tf.reduce_mean(loss_vector) * loss_vector)
        
        return tf.reduce_mean(loss_vector) 
        
        


from tensorflow.keras.applications import MobileNet, ResNet50
from tensorflow.keras.optimizers import Adam

dset = CalebA(True, False)
train_dset, val_dset = dset.get_dset('train'), dset.get_dset('val')
m = ResNet50(classes = 40,weights = None,input_shape=[218,178,3])
w = ResNet50().get_weights()
w[-1] = w[-1][:40]
w[-2] = w[-2][:,:40]
m.set_weights(w)

m.compile(optimizer = Adam(0.001), loss = 'binary_crossentropy', metrics = 'accuracy')


h = m.fit(train_dset, validation_data = val_dset, epochs = 20)
m.save_weights('resnet_weights.h5')

np.savetxt('loss.txt',h)


