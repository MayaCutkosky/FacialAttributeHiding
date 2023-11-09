#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 09:43:43 2023

@author: maya
"""

import numpy as np
from torch import empty, float32, from_numpy
from torch.utils.data import Dataset
from torchvision.io import read_image
from threading import Thread
from torchvision import transforms

class CelebA(Dataset):
    def __init__(self, attributes=True, identities=False, batch_size = 32,
                 cuda = True, directory = '/home/maya/Desktop/datasets/Celeba/', attr_format='0,1'):
        '''
        

        Parameters
        ----------
        attributes : TYPE, list of ints or bool
            DESCRIPTION. The default is True.
        identities : TYPE, Bool
            DESCRIPTION. Currently does nothing. Option will be added later
        batch_size : TYPE, int
            DESCRIPTION. The default is 32.
        cuda : TYPE, bool
            DESCRIPTION. The default is True.
        directory : TYPE, string
            DESCRIPTION. The default is '/home/maya/Desktop/datasets/Celeba/'.

        Returns
        -------
        None.

        '''
        
        self.identities = identities
        self.input_shape = [3,224, 224]
        self.batch_size = batch_size
        self.dir = directory
        self.cuda = cuda
        
        self.num_threads = 8
        #if self.cuda:
        #    self.preloaded_images = empty(size = [batch_size]+self.input_shape, dtype = float32).cuda()
        #self.preload_dset()
        
        
        if identities:
            self.ids = np.genfromtxt(self.dir + 'identity_CelebA.txt')
        
        if type(attributes) == bool:
            if attributes:
                attributes = np.arange(1,41)
            else:
                attributes = []
            
        if len(attributes):
            self.attr = from_numpy(np.genfromtxt(self.dir + 'list_attr_celeba.txt',skip_header=2, usecols=attributes))
            if attr_format == '0,1':
                self.attr = (self.attr+ 1) // 2
            
            

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
        return self.get_sample(idx)

    def preload_dset(self,idx=None):
        if idx == None:
            idx = np.random.choice(np.arange(1,202599),self.batch_size,False)
        self.preloaded_idx = idx
        #load images
        self.threads = []
        for i, idx_lst in enumerate(np.array_split(idx, self.num_threads)):
            Thread(target=self.threaded_load_image,args= [idx_lst,i]).start()
        
            
        
    
    def threaded_load_image(self, idx_list, sample_num):

        for i,j in enumerate(idx_list):
            self.preloaded_images[sample_num*self.batch_size//self.num_threads + i] = self.load_image(j)
        
        
        
    def load_image(self,idx):
        idx = str(idx).zfill(6)
        im = read_image(self.dir + 'img_align_celeba/'  + idx + '.jpg').cuda()/255
        im = transforms.Compose((
            transforms.CenterCrop([170,170]),
            transforms.Resize(size=(224, 224), antialias=True)
            ))(im)
        return im
    
    
    def get_sample(self, idx = None, attr = False, identity = False, partition = 'train'):
        if idx == None:
            idx = np.random.choice(np.arange(1,202599),self.batch_size,False)
        sample = empty(size = [self.batch_size]+self.input_shape, dtype = float32, requires_grad=True).cuda()
        for i,j in enumerate(idx):
            sample[i] = self.load_image(j+1)
        sample = [sample]
        if attr:
            sample.append(self.attr[idx].cuda())
        if identity:
            sample.append(self.identity[self.preloaded_idx])
        
        #self.preload_dset()
        
        return tuple(sample)

        

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

if __name__ == '__main__':
    dset = CelebA()
    
