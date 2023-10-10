#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 09:43:43 2023

@author: maya
"""

import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image


class CelebA(Dataset):
    def __init__(self, attributes=True, identities=True, batch_size = 32,
                 cuda = True, directory = '/home/maya/Desktop/datasets/Celeba/'):
        
        self.identities = identities
        self.input_shape = [224, 224, 3]
        self.batch_size = batch_size
        self.dir = directory
        self.cuda = cuda
        
        self.num_threads = 8
        self.preloaded_images = torch.empty(size = [batch_size]+self.input_shape, dtype = torch.float32)
        self.preload_images()
        
        
        if identities:
            self.ids = np.genfromtxt(self.dir + 'identity_CelebA.txt')
        
        if type(attributes) == bool:
            if attributes:
                attributes = np.arange(1,41)
            else:
                attributes = []
            
        if len(attributes):
            self.attr = (np.genfromtxt(self.dir + 'list_attr_celeba.txt',skip_header=2, usecols=attributes)) + 1) // 2

        
    def __get_next__(self,idx):
        '''Used to get images for images

        Parameters
        ----------
        idx : int or list of ints
            

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return get_sample(idx)

    def preload_images(self,idx):
        #load images
        self.threads = []
        for idx_lst in np.array_split(range(self.batch_size), self.num_threads):
            Thread(target=self.theaded_load_image,args= [idx_list])
            
        
    
    def threaded_load_image(self, idx_list):
        for i in idx_list:
            self.preloaded_images[i] = self.load_image(i)
        
        
    def load_image(self,idx):
        idx = str(idx).zfill(6)
        im = read_image(self.dir + 'img_align_celeba/'  + idx + '.jpg')
        return im.cuda() 
    
    
    def get_sample(self, idx = None, attr = False, identity = False, partition = 'train'):
        if idx == None:
            
            
        self.preload_dset()
        return self.preloaded_images

        
        
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
        
        dset 

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
