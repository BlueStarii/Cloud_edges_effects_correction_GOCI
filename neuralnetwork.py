# -*- coding: utf-8 -*-
"""
Created on Sat May 22 11:33:36 2021

@author: Administrator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self,n_input=11,n_hidden1=300,n_hidden2=75,n_hidden3=38,n_hidden4=18,n_output=6):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,1,kernel_size=(1,11),stride=11)
        self.hidden1 = nn.Linear(n_input,n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1,n_hidden2)
        self.hidden3 = nn.Linear(n_hidden2,n_hidden3)
        self.hidden4 = nn.Linear(n_hidden3,n_hidden4)
        self.predict = nn.Linear(n_hidden4,n_output)
    def forward(self,input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.relu(out)
        out = self.hidden3(out)
        out = F.relu(out)
        out = self.hidden4(out)
        out = F.relu(out)
        out = self.predict(out)

        return out