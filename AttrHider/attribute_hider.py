#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:15:57 2023

@author: mkcc68


Dataset:
    (image, attributes, identity)
Generator:
    input = input Image or nothing
    Output = Image that looks similar to input image
Classifier:
    input: Image
    Output:
        input image -> correct attr
        protected image -> incorrect attr
Identifier: 
    Input: Image
    Output: (correct) identity
    Probably frozen.
    
"""
# from sys import path as syspath
# from os.path import abspath
# path = abspath('..')
# if path not in syspath:
#     syspath.append(path)
    
# from Dataset_loaders.datasets_torch import CelebA
from torchvision.datasets import CelebA
from torchvision import transforms
from res_facenet.models import model_921 as facenet
from torch import nn
import torch
from itertools import chain

class Metric:
    def __init__(self, scoring_fun = lambda x: x):
        self.running_score = 0
        self.scoring_fun = scoring_fun
    def update(self, *args):
        score = self.scoring_fun(*args)
        self.running_score = self.running_score * 0.8 + score * 0.2
    def output(self):
        return self.running_score

import numpy as np

class AttrHider():
    
    @staticmethod
    def Encoder():
        return nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, padding = 1 ),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, 4, 2, padding = 1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(128, 256, 4, 2, padding = 1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256, 512, 4, 2, padding = 1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(),
                nn.Conv2d(512, 1024, 4, 2, padding = 1),
                nn.InstanceNorm2d(1024),
                nn.LeakyReLU(),
                nn.MaxPool2d(7),
                nn.Flatten(),
                nn.Linear(1024,1)
            )
    
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_layers = nn.ModuleList()
            self.decoder_layers = nn.ModuleList()
            channels = [3, 64, 128, 256, 512, 1024]
            for i in range(5):
                
                self.encoder_layers.append(nn.Sequential(
                        nn.Conv2d(channels[i], channels[i+1], 4, 2, padding = 1),
                        nn.InstanceNorm2d(channels[i+1]),
                        nn.LeakyReLU()
                    ))
                self.decoder_layers.append(nn.Sequential(
                        nn.ConvTranspose2d(channels[-i-1],channels[-i-2], 4, 2, padding = 1),
                        nn.InstanceNorm2d(channels[-i-2]),
                        nn.LeakyReLU()
                    ))
            
            
        def forward(self, x):
            for l in self.encoder_layers:
                x = l(x)
            x = torch.rand(len(x),1024,7,7, device = 'cuda') - x
            for l in self.decoder_layers:
                x = l(x)
            return x
    
    def __init__(self):
        self.classifier = self.Encoder()
        # self.classifier.add_module('C_maxpool',nn.MaxPool2d(7))
        # self.classifier.add_module('C_flatten', nn.Flatten())
        # self.classifier.add_module('C_linear', nn.Linear(1024,1))
        self.classifier = self.classifier.cuda()
        self.detector = self.Encoder()
        # self.detector.add_module('D_maxpool',nn.MaxPool2d(7))
        # self.detector.add_module('D_flatten', nn.Flatten())
        # self.detector.add_module('D_linear', nn.Linear(1024,1))
        self.detector = self.detector.cuda() #1 if real, 0 if generated
        self.generator = self.Generator().cuda()
        self.identifier = facenet().requires_grad_(False).cuda()
        self.optimizer_D = torch.optim.Adam(chain(self.detector.parameters(),self.classifier.parameters()),
                                            lr = 0.02, betas=(0.5,0.99) )
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr = 0.02,betas=(0.5,0.99))
        
        transform = transforms.Compose((transforms.ToTensor(),
            transforms.CenterCrop([170,170]),
            transforms.Resize(size=(224, 224), antialias=True), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ))
        dataset = CelebA('/home/maya/Desktop/datasets/',transform=transform)
        
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, num_workers = 2, shuffle = True)
        self.G_loss = Metric()
        self.D_loss = Metric()
        self.loss_fn = nn.BCEWithLogitsLoss()
        def acc(y_true, y_pred): 
            return np.mean(y_true==(y_pred>0.5))  
        self.acc_metric = Metric(acc)
        self.hidden_acc_metric = Metric(acc)

    def train_step(self,n=5):        
         
        def calc_gradient(network, orig_images, protected_images):
            epsilon = torch.tile(torch.rand(len(orig_images),device = 'cuda'),[224,224,3,1]).T
            merged_images = epsilon * orig_images + (1-epsilon) * protected_images
            network(merged_images).backward(gradient = torch.ones([len(merged_images),1],device = 'cuda'),inputs = merged_images)
            return merged_images.grad


        for i, (x,y) in enumerate(self.dataloader):
            self.optimizer_D.zero_grad()
            

            orig_images = x.to('cuda')
            classifications = y.type(torch.float)[:,6].to('cuda')
            if i > n:
                break
            protected_images = self.generator(orig_images)
            

            
            grad_x = calc_gradient(self.detector, orig_images, protected_images)
            #grad_x += calc_gradient(classify, orig_images, protected_images)
            gradient_penalty = 10 * torch.square(torch.norm(grad_x) - 1)               
            
            
            loss = self.detector(protected_images) - self.detector(orig_images) 
            
            loss += gradient_penalty
            ind_has_attr = classifications == 1
            num_with_attr = int(classifications.sum().detach())
            
            orig_images_attr = self.classifier(orig_images).flatten()
            protected_images_attr = self.classifier(protected_images).flatten()
            attr_loss = 4.15289536 * self.loss_fn(protected_images_attr[ind_has_attr], torch.ones((num_with_attr),device='cuda')) 
            attr_loss += 4.15289536 * self.loss_fn(orig_images_attr[ind_has_attr], torch.ones((num_with_attr),device='cuda'))
            attr_loss += 1.31716879 * self.loss_fn(protected_images_attr[~ind_has_attr], torch.zeros((len(ind_has_attr) - num_with_attr),device='cuda')) 
            attr_loss += 1.31716879 * self.loss_fn(orig_images_attr[~ind_has_attr], torch.zeros((len(ind_has_attr) - num_with_attr),device='cuda'))
 
            

            loss = torch.mean(loss)
            loss += attr_loss
            
            loss.backward()
            self.optimizer_D.step()
            self.D_loss.update(loss.detach().cpu().numpy())
            
        
        self.optimizer_G.zero_grad()
        
        
        protected_images = self.generator(orig_images)
        protected_images_id_features = self.identifier(protected_images)
        orig_images_id_features = self.identifier(orig_images)
        protected_image_classifications = self.classifier(protected_images).flatten()
        
        loss = - self.detector(protected_images) - self.loss_fn(protected_image_classifications, classifications)
        loss = torch.mean(loss)
#        loss += -nn.functional.binary_cross_entropy_with_logits(protected_images_attr, classifications)
        loss += nn.functional.mse_loss(protected_images_id_features, orig_images_id_features)
        
        loss.backward()
        #self.optimizer_G.step()
        self.G_loss.update(loss.detach().cpu().numpy())
        self.acc_metric.update( classifications.cpu().numpy(), self.classifier(orig_images).flatten().detach().cpu().numpy())
        self.hidden_acc_metric.update( classifications.cpu().numpy(), protected_image_classifications.detach().cpu().numpy())
    


    def train(self, steps = 100000, verbose = True,**kwargs):
        for i in range(steps):
            self.train_step(**kwargs)
            if verbose:
                print('i Discriminator Loss   Generator Loss  AccOrig AccProtected')
                if i%100 == 0:
                    print(i, self.D_loss.output(), self.G_loss.output(), self.acc_metric.output(), self.hidden_acc_metric.output())
                




if __name__ == '__main__':
    m = AttrHider()
    m.train()
