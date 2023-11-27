#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:27:48 2023

@author: mkcc68
"""

from torchvision.models import resnet50
from torch import nn, optim
import torch

def make_model():
    return resnet50(num_classes = 40)
    
    
def train_model(m,dset):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = m.parameters())
    for x,y in dset:
        x = x.cuda()
        y = y.cuda()
        
        optimizer.zero_grad()
        l = loss(m(x),y)
        l.backward()
        optimizer.step()
    

class Metric:
    def __init__(self,fun):
        self.total = 0
        self.correct = 0
        self.fun = fun
    def add(self,**kwargs):
        self.fun(**kwargs)
    def result(self):
        return self.correct/self.total
        

def evaluate_model(m,dset, transformation = lambda x: x):
    ave = Metric(lambda x,y : torch.mean(x-y))
    ave40 = Metric(lambda x,y : torch.mean(x[:,40]-y[:,40]))
    ave20 = Metric(lambda x,y : torch.mean(x[:,20]-y[:,20]))
    ave26 = Metric(lambda x,y : torch.mean(x[:,26]-y[:,26]))
    for x,y in dset:
        x = x.cuda()
        x = transformation(x)
        ytrue = y.cuda()
        ypred = (m(x) > 0.5).type(torch.float)
        
        ave20.add(ytrue,ypred)
        ave26.add(ytrue,ypred)
        ave40.add(ytrue,ypred)
        ave.add(ytrue,ypred)
    print('Average: ', ave.result())
    print('Age', ave40.result())
    print('Gender', ave20.result())
    print('Skin color', ave26.result())
    

    
        
        
        
