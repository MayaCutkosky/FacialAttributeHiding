#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 09:43:43 2023

@author: maya
"""

import numpy as np
from tensorflow.data import Dataset
from tensorflow.io import read_file, decode_jpeg
from tensorflow import cast, float32
from tensorflow.image import resize
from dlib import get_frontal_face_detector
detector = get_frontal_face_detector()
class CalebA():
    def __init__(self, attributes, identities, directory = '/home/maya/Desktop/datasets/Celeba/'):
        self.identities = identities
        self.attr = attributes
        self.input_shape = [224,224, 3]
        self.dir = directory
        
    def load_attr(self, start=0, end=0):
        return (np.genfromtxt(self.dir + 'list_attr_celeba.txt',skip_header=2+start, skip_footer = end, usecols=np.arange(1,41)) + 1) // 2

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
            attr = self.load_attr(start,end)[random_shuffle]
            
        #load image
        def load_image(im):
            f = read_file(self.dir + 'img_align_celeba/'  + im + '.jpg')
            im = decode_jpeg(f)
            return im
        
        
        def center_and_resize_image(im):
            # dim_x, dim_y = data[1]
            # crop_size = ((dim_x - dim_y))//2
            im = im[24:194,4:174]
            im = resize(im, self.input_shape[:2])
            return im
        
        #create dataset
        x_dset = Dataset.from_tensor_slices(im_files)
        x_dset = x_dset.map(load_image)
        
        x_dset = x_dset.map(center_and_resize_image)
        
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

attr_weights = [ 7.99800142,  2.74559068,  0.95120049,  3.88826425, 43.55663075,
        5.59738188,  3.15289536,  3.26380588,  3.179712  ,  5.75712904,
       18.64691621,  3.87344847,  6.0339548 , 16.37108806, 20.41864891,
       14.35655272, 14.93260459, 22.83798094,  1.58450057,  1.19764831,
        1.39949546,  1.06856099, 23.07021504,  7.68442711,  0.19769092,
        2.51936005, 22.28456499,  2.60432307, 11.53473984, 14.21584679,
       16.69578129,  1.07434293,  3.79842262,  2.12923205,  4.29310795,
       19.63546547,  1.11669017,  7.13226027, 12.7523079 ,  0.29262955]
