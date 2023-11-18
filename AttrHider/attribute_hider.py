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
    
    
        

Steps:
    1. train classifier [1,0,0,0,0,0,0]
    2. train generator/disciminator [0,0,1,10,1,0,0]

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
from torch.utils.tensorboard import SummaryWriter

from os.path import exists as pathexists, join as pathjoin
from os import makedirs,rmdir

class Metric:
    def __init__(self, scoring_fun = lambda x: x):
        self.running_score = 0
        self.scoring_fun = scoring_fun
    def update(self, *args):
        score = self.scoring_fun(*args)
        self.running_score = self.running_score * 0.7 + score * 0.3
    def output(self):
        return self.running_score

import numpy as np

class AttrHider():
    
    @staticmethod
    def Encoder():
        return nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, padding = 1 ),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.3),
                nn.Conv2d(64, 128, 4, 2, padding = 1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.3),
                nn.Conv2d(128, 256, 4, 2, padding = 1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.3),
                nn.Conv2d(256, 512, 4, 2, padding = 1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.3),
                nn.Conv2d(512, 1024, 4, 2, padding = 0),
                nn.InstanceNorm2d(1024),
                nn.LeakyReLU(0.3),
                nn.MaxPool2d(6),
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
                if i == 4:
                    pad = 0
                else:
                    pad = 1
                self.encoder_layers.append(nn.Sequential(
                        nn.Conv2d(channels[i], channels[i+1], 4, 2, padding = pad),
                        nn.InstanceNorm2d(channels[i+1]),
                        nn.LeakyReLU()
                    ))
                if i == 0:
                    pad = 0
                else:
                    pad = 1
                self.decoder_layers.append(nn.ModuleList())
                self.decoder_layers[-1].append(
                        nn.ConvTranspose2d(channels[-i-1],channels[-i-2], 4, 2, padding = pad)
                    )
                if i < 3:
                    self.decoder_layers[-1].append(
                         nn.Sequential(nn.Conv2d(channels[-i-2]*2,channels[-i-2], 3, padding = 1),
                                      nn.InstanceNorm2d(channels[-i-2]),
                                      nn.LeakyReLU()
                                      )
                         )
            self.norm_layer = nn.Sequential(
                    nn.Flatten(),
                    nn.BatchNorm1d(1024*6*6, affine=False),
                    nn.Unflatten(1,(1024,6,6))
                )

            
        def forward(self, x, r = None):
            if r is None:
                r = torch.rand(len(x),1024,6,6, device = 'cuda') 
            encoder_inputs = [] #For skip connections
            for l in self.encoder_layers:
                encoder_inputs.append(x)
                x = l(x)
            #set x between 0 and 1
            x = self.norm_layer(x)
            x = torch.abs(x - r )
            for l in self.decoder_layers:
                x = l[0](x)
                if len(l) == 2:
                    x = torch.concatenate([x,encoder_inputs.pop()],1)
                    x = l[1](x)
            return x
    
    def __init__(self, savedir = 'Output', attr_id = -1):
        '''
        

        Parameters
        ----------
        savedir : str, optional
            Directory that checkpoint files and tensorboard info is saved to. The default is 'Output'.
        attr_id : int, optional
            Attribute that is being loaded. The default is -1.

        Returns
        -------
        AttrHider object (for training attribute hider)

        '''
        
        #Build networks
        self.classifier = self.Encoder()
        self.classifier = self.classifier.cuda()
        self.discriminator = self.Encoder()
        self.discriminator = self.discriminator.cuda() #1 if real, 0 if generated
        self.generator = self.Generator().cuda()
        self.identifier = facenet().requires_grad_(False).cuda()
        self.optimizer_D = torch.optim.Adam(chain(self.discriminator.parameters(),self.classifier.parameters()),
                                            lr = 0.002, betas=(0.5,0.99) )
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr = 0.02,betas=(0.5,0.99))
        
        #Training hyperparameters
        self.gradient_coeff = 0
        self.classifier_coeff = 1
        self.protected_image_classifier_coeff = 0
        self.discriminator_coeff = 0
        self.generator_vs_discriminator_coeff = 0
        self.generator_vs_classifier_coeff = 0
        self.id_coeff = 0
        self.im_coeff = 0.01
        
        
        #Get dataset
        transform = transforms.Compose((transforms.ToTensor(),
            transforms.CenterCrop([170,170]),
            transforms.Resize(size=(224, 224), antialias=True), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ))
        dataset = CelebA('/home/maya/Desktop/datasets/',transform=transform)
        
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, num_workers = 2, shuffle = True)

        #Build Metrics
        self.G_loss = Metric()
        self.D_loss = Metric()
        self.d_loss_fn = None
        self.g_loss_fn = None
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.identity_loss_fn = nn.CosineEmbeddingLoss()
        def acc(y_true, y_pred): 
            return np.mean(y_true==(y_pred>0))  
        self.acc_metric = Metric(acc)
        self.hidden_acc_metric = Metric(acc)
        self.im_loss_fn = nn.L1Loss()
        #Set up/load from save directory
        self.savedir = savedir
        
        #set up tensorboard
        self._step = 0 
        self.writer = SummaryWriter(savedir)
        
        
        #Other variables
        self._attr_id = attr_id
        
        
    def get_config(self):
        return {
                'attr_id' : self._attr_id                
            }
    
    def _zero_grad(self):
        self.classifier.zero_grad()
        self.discriminator.zero_grad()
        self.generator.zero_grad()
    
    def train_step(self,n=6):

        
        def calc_gradient(network, orig_images, protected_images):
            epsilon = torch.rand(len(orig_images),1,1,1,device = 'cuda').expand(-1,3,224,224)
            merged_images = epsilon * orig_images + (1-epsilon) * protected_images
            network(merged_images).backward(gradient = torch.ones([len(merged_images),1],device = 'cuda'),inputs = merged_images)
            return merged_images.grad

        for i, (x,y) in enumerate(self.dataloader):
            self._zero_grad()
            

            orig_images = x.to('cuda')
            classifications = y.type(torch.float)[:,self._attr_id].to('cuda')
            if i >= n:
                break
            protected_images = self.generator(orig_images)
            
            if self.d_loss_fn is None:

                loss = 0
                grad_x = calc_gradient(self.discriminator, orig_images, protected_images)
                gradient_penalty = self.gradient_coeff * torch.square(torch.norm(grad_x,2, dim=1) - 1).mean()               
                
                    
                disc_loss = self.discriminator(protected_images) - self.discriminator(orig_images) 
                
                disc_loss += gradient_penalty
                disc_loss = disc_loss.mean()
                loss += self.discriminator_coeff * disc_loss
                
                
                ind_has_attr = classifications == 1
                num_with_attr = int(classifications.sum().detach())
                if num_with_attr > 0  and num_with_attr < 32:
                    orig_images_attr = self.classifier(orig_images).flatten()
                    protected_images_attr = self.classifier(protected_images).flatten()
                    attr_loss = self.protected_image_classifier_coeff*0.4527663019067221 * self.loss_fn(protected_images_attr[ind_has_attr], torch.ones((num_with_attr),device='cuda')) 
                    attr_loss += 0.4527663019067221* self.loss_fn(orig_images_attr[ind_has_attr], torch.ones((num_with_attr),device='cuda'))
                    attr_loss += self.protected_image_classifier_coeff*0.5472336980932778 * self.loss_fn(protected_images_attr[~ind_has_attr], torch.zeros((len(ind_has_attr) - num_with_attr),device='cuda')) 
                    attr_loss += 0.5472336980932778* self.loss_fn(orig_images_attr[~ind_has_attr], torch.zeros((len(ind_has_attr) - num_with_attr),device='cuda'))
                    loss += self.classifier_coeff * attr_loss
                else:
                    continue
            else:
                loss = self.d_loss_fn(orig_images, protected_images, classifications)

                
            loss.backward()
            self.optimizer_D.step()
            self.D_loss.update(loss.detach().cpu().numpy())
            
            #update writer
            self.writer.add_scalar('loss_discriminator', disc_loss.detach().cpu().numpy(), self._step)
            self.writer.add_scalar('loss_classifier', attr_loss.detach().cpu().numpy(), self._step)
            self._step += 1
                
            if self._step%10 == 0: #update every 10 steps
                self.writer.add_histogram('classifier_output', orig_images_attr,self._step)
                self.writer.add_histogram('current_classifier_output', orig_images_attr)
            if self._step%100 == 0: #update every 100 steps
                self.writer.add_image('generated_image', protected_images[0],self._step)
                self.writer.add_image('original_image', orig_images[0],self._step)
            
        
        self._zero_grad()
        
        
        protected_images = self.generator(orig_images)
        protected_images_id_features = self.identifier(protected_images)
        orig_images_id_features = self.identifier(orig_images)
        protected_image_classifications = self.classifier(protected_images).flatten()
        
        if self.g_loss_fn is None:
            G_loss = - self.discriminator(protected_images).mean() 
            attr_loss = - self.loss_fn(protected_image_classifications, classifications)
            im_loss = self.im_loss_fn(orig_images,protected_images)
            id_loss = self.identity_loss_fn(protected_images_id_features, orig_images_id_features,torch.ones(1).cuda())
            loss = self.generator_vs_discriminator_coeff * G_loss + self.generator_vs_classifier_coeff * attr_loss + self.id_coeff * id_loss + self.im_coeff * im_loss
        else:
            loss = self.g_loss_fn(orig_images, protected_images, classifications, protected_images_id_features, orig_images_id_features, protected_image_classifications)

        loss.backward()
        self.optimizer_G.step()
        self.G_loss.update(loss.detach().cpu().numpy())
        self.acc_metric.update( classifications.cpu().numpy(), self.classifier(orig_images).flatten().detach().cpu().numpy())
        self.hidden_acc_metric.update( classifications.cpu().numpy(), protected_image_classifications.detach().cpu().numpy())
        
        #update writer
        self.writer.add_scalar('loss_confuse_discriminator', G_loss.detach().cpu().numpy(), self._step)
        self.writer.add_scalar('loss_confuse_classifier', attr_loss.detach().cpu().numpy(), self._step)
        self.writer.add_scalar('loss_id', id_loss.detach().cpu().numpy(), self._step)
        
    
    def train(self, steps = 100000, verbose = True, save_chkpt_freq = 0, **kwargs):
        '''
        

        Parameters
        ----------
        steps : int, optional
            Number of iterations that one trains. The default is 100000.
        verbose : bool, optional
            Whether to print out parameters. The default is True.
        save_chkpt_freq : TYPE, optional
            How often to save checkpoint files. The default value (0) represents no saving checkpoint files.
        **kwargs : 
            Parameters to feed to self.train_step

        Returns
        -------
        None.

        '''
        if verbose:
            print('i Discriminator Loss   Generator Loss  AccOrig AccProtected')
        for i in range(steps):
            self.train_step(**kwargs)
            if save_chkpt_freq:
                if i%save_chkpt_freq == 0:
                    self.save(str(self._step)+'.pt')
                    
            if i%100 == 0:
                if verbose:
                    print(i, self.D_loss.output(), self.G_loss.output(), self.acc_metric.output(), self.hidden_acc_metric.output())


    def change_coeff(self, coeffs):
        self.classifier_coeff = coeffs[0]
        self.protected_image_classifier_coeff = coeffs[1]
        self.discriminator_coeff = coeffs[2]
        self.gradient_coeff = coeffs[3]
        self.generator_vs_discriminator_coeff = coeffs[4]
        self.generator_vs_classifier_coeff = coeffs[5]
        self.id_coeff = coeffs[6]
    
    def save(self, filename=None):
        if filename is None:
            filename = 'GAN.pt'
        torch.save([
                self.classifier.state_dict(),
                self.discriminator.state_dict(),
                self.generator.state_dict(),
                self.optimizer_D.state_dict(),
                self.optimizer_G.state_dict()
            ], pathjoin(self.savedir, filename) )
    
    def load(self, filename=None):
        params = torch.load(filename)
        self.optimizer_G.load_state_dict(params.pop())
        self.optimizer_D.load_state_dict(params.pop())
        self.generator.load_state_dict(params.pop())
        self.discriminator.load_state_dict(params.pop())
        self.classifier.load_state_dict(params.pop())
        

    def train_generator(self):
        
        csd = self.classifier.state_dict()
        for i,l in enumerate(self.generator.encoder_layers):
            sd = l.state_dict()
            for key in sd.keys():
                sd[key] = csd[str(i*3)+key[1:]]
            l.load_state_dict(sd)
            
        optim = torch.optim.Adam(chain(*[l.parameters() for l in m.generator.decoder_layers]), lr = 0.02,betas=(0.5,0.99))
        self._zero_grad()
        for x,_ in m.dataloader:
            optim.zero_grad()
            x = x.to('cuda')
            y = m.generator(x)
            l = torch.abs(y - x).mean()
            l.backward()
            optim.step()
            
            if self._step%100 == 0:
                print(self._step,l)
                self.writer.add_scalar('image difference', l, self._step)
                self.writer.add_image('generated_image', y[0],self._step)
                self.writer.add_image('original_image', x[0],self._step)
            self._step = self._step + 1

if __name__ == '__main__':
    m = AttrHider()
    #Train classifier
    #m.change_coeff([1,0,0,0,0,0,0])
    #m.train(5000)
    #Try to get a realistic image
    #m.train_generator()
    #m.save('test.pt')
    
    # #    The discriminator trains much better than the generator. How to solve this? Decrease n, Increase gradient penalty? Decrease discriminator loss (Increasing generator loss will likely not help)
    # m.change_coeff([0,0,1,10,1,0,1])
    # m.train()
