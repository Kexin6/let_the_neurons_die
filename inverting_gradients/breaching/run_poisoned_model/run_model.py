
from __future__ import print_function
import argparse, textwrap
import copy, math
from unittest import result
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import custom_model
import adversary

class runModel():
    def __init__(self, image_size):
        torch.manual_seed(1)
        self.device = torch.device("cuda")
        self.temp_model = custom_model.DenseNet(image_size).to(self.device)
        self.optimizer = optim.Adadelta(self.temp_model.parameters(), lr=1.0)
    
    def train(self, data, target, index):
        self.temp_model.train()
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.temp_model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.optimizer.step()

        attack_weights = np.empty([0]) # weights of adversary
        for layer in self.temp_model.state_dict():
            if 'weight' in layer:
                attack_weights = np.concatenate((attack_weights, self.temp_model.state_dict()[layer].data.cpu().detach().numpy()), axis=None)
        # print("Knocked out weight")
        return attack_weights[index]

    def run_temp(self, temp_loader, index):
        for _, (data, target) in enumerate(temp_loader):
            value = self.train(data, target, index)
        return value

    
            
    