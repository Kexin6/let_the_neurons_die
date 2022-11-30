
from __future__ import print_function
import argparse, textwrap
import copy, math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import custom_model


class runModel():
    def __init__(self, image_size):
        torch.manual_seed(1)
        self.device = torch.device("cuda")
        self.temp_model = custom_model.DenseNet(image_size).to(self.device)
        self.temp_model.load_state_dict(torch.load("run_4.pt"))
        self.temp_model.eval()
        self.optimizer = optim.Adadelta(self.temp_model.parameters(), lr=1.0)
        
    
    def train(self, data, target, indexes):
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

        score = 0
        for index in indexes:
            score += abs(attack_weights[index])
        point = len(indexes) + 1
        last_point = indexes[len(indexes) -1]
        if abs(attack_weights[last_point]) < 0.003:
            point = int(len(indexes) + 2)
        
        return score, point

    def run_temp(self, temp_loader, indexes):
        for _, (data, target) in enumerate(temp_loader):
            value, point = self.train(data, target, indexes)
        return value, point

    def train2(self, data, target):
        self.temp_model.train()
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.temp_model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.optimizer.step()
        attack_weights = self.temp_model.state_dict()['model_arch.5.weight'].data.cpu().detach().numpy()
        return np.sum(attack_weights)

    def run_temp2(self, temp_loader):
        for _, (data, target) in enumerate(temp_loader):
            value = self.train2(data, target)
        return value

    
            
    