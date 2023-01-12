
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import *


class attackModelRunner():
    def __init__(self, args, image_size):
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda")
        if args.model == "MNISTNet":
            self.temp_model = MNISTNet(image_size).to(self.device)
        
        self.temp_model.load_state_dict(torch.load(args.candidate_pt))
        self.temp_model.eval()
        self.optimizer = optim.Adadelta(self.temp_model.parameters(), lr=args.lr)

    def train_attack(self, data, target, attack_layer):
        self.temp_model.train()
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.temp_model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.optimizer.step()
        attack_weights = self.temp_model.state_dict()[attack_layer].data.cpu().detach().numpy()
        return np.sum(attack_weights)

    def run_attack(self, temp_loader, attack_layer):
        for _, (data, target) in enumerate(temp_loader):
            value = self.train_attack(data, target, attack_layer)
        return value
            
    