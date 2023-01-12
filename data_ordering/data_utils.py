# Helper functions for reading, and writing data for models
import torch, csv
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from CustomSampler import CustomSampler
from torch.utils.data import DataLoader
from models import *

def load_candidates(file_name):
    print('Fetching Candidate List: ' + file_name)
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        candidate_list = list(reader)
    candidate_list = candidate_list[0]
    candidate_list = [int(i) for i in candidate_list]
    return candidate_list

def save_candidates(file_name, candidate_list):
    print('Saving Candidate List: ' + file_name)
    with open(file_name, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(candidate_list)

def save_model_candidates(args, image_size, device, train_dataset, candidate_list, attack_layer, save_name):
    if args.model == "MNISTNet":
        model = MNISTNet(image_size).to(device)
    
    sampler = CustomSampler(candidate_list[:-1], candidate_list[-1:][0])
    loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=1, sampler=sampler)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    simple_train(device, model, loader, optimizer)

    attack_weights = model.state_dict()[attack_layer].data.cpu().detach().numpy()
    print('Weights: ' + str(np.sum(attack_weights)))
    torch.save(model.state_dict(), save_name)

def simple_train(device, model, loader, optimizer):
    model.train()
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()