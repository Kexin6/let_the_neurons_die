from __future__ import print_function
import argparse
import copy, math
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
import model

# Algorithm
# Run the temp_model and adversarial_model on the datapoint
# Calculate their updates
# Compare their weight deltas (do we need to do a baseline compare or is gradient enough)
# Calculate the score based on the gradient
def score_datapoint_weights(args, temp_model, adversarial_model, device, train_loader):
    base_model = copy.deepcopy(temp_model.state_dict())
    weights = []
    score_list = {}

    for param in temp_model.parameters():
        weights.append(param.clone())

    adversarial_weights = np.empty([0]) # weights of adversary
    for param in adversarial_model.parameters():
        adversarial_weights = np.concatenate((adversarial_weights, param.cpu().detach().numpy()), axis=None)
    
    i = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Set model
        if i % 1000 == 0:
            print("On data point: " + str(i))

        temp_model.load_state_dict(base_model)
        temp_model.train()
        temp_optimizer = optim.Adadelta(temp_model.parameters(), lr=args.lr)
        data, target = data.to(device), target.to(device)

        # Temp Model
        temp_optimizer.zero_grad()
        output_temp = temp_model(data)
        loss_temp = F.nll_loss(output_temp, target)
        loss_temp.backward()
        temp_optimizer.step()

        temp_weights = np.empty([0]) # weights after backprop
        for param in temp_model.parameters():
            temp_weights = np.concatenate((temp_weights, param.cpu().detach().numpy()), axis=None)
    
        gradient_delta = np.abs(np.subtract(temp_weights, adversarial_weights))
        score_np = np.sum(gradient_delta)
        score_list[i] = score_np
        i+= 1
    
    return score_list

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def get_weights(model):
    weights = []
    for param in model.parameters():
        weights.append(param.view(-1).clone())

    weights = torch.cat(weights)
    return weights
    
def Merge(dict1, dict2):
    return(dict2.update(dict1))

def load_attacker(weight_file, network):
    saved = np.load(weight_file, allow_pickle=True)
    for elem in saved:
        print(np.shape(elem))
    print("Adversarial Model Shape")
    i = 0
    for layer in network.state_dict():
        print(layer)
        if 'weight' in layer: 
            print(np.shape(saved[i]))
            transposed = np.transpose(saved[i])
            new_weights = torch.from_numpy(transposed)
            network.state_dict()[layer].data.copy_(new_weights)
            i += 1
    return network

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Data Ordering Attack')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=7, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--attack-type', type=str, default='single',
                        help='Pick attack type')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

    # In order dataset
    dataset_size = len(dataset1)
    dataset_indices = list(range(dataset_size))
    train_sampler = SequentialSampler(dataset_indices)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
   

    # Define Models

    adversarial_model = model.DenseNet().to(device)
    temp_model = model.DenseNet().to(device)
    attack_model = model.DenseNet().to(device)

    weight_file = 'weights_target_array.npy'
    adversarial_model = load_attacker(weight_file, adversarial_model)

    attack_optimizer = optim.Adadelta(attack_model.parameters(), lr=args.lr)
    # Create a sampler for the attack
    attack_sampler = SequentialSampler(dataset_indices)
    
    # Specific attacks
    if args.attack_type == 'single':
        # Run through all the datapoints once to get score and do it in one shot
        print("Running simple data order attack")
        temp_loader = DataLoader(dataset=dataset1, shuffle=False, batch_size=1, sampler=train_sampler)
        score_dict = score_datapoint_weights(args, temp_model, adversarial_model, device, temp_loader)
        
        # Higher the score the worse it is
        sorted_scores = sorted(score_dict, key=score_dict.get) 
        attack_sampler = SequentialSampler(sorted_scores)

        attack_loader = DataLoader(dataset=dataset1, shuffle=False, batch_size=200, sampler=attack_sampler)
        scheduler = StepLR(attack_optimizer, step_size=1, gamma=args.gamma)

        for epoch in range(1, args.epochs + 1):
            train(args, attack_model, device, attack_loader, attack_optimizer, epoch)
            test(attack_model, device, test_loader)
            scheduler.step()
    elif args.attack_type == 'every_epoch':
        print("Running every epoch")
        scheduler = StepLR(attack_optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            attack_weights = copy.deepcopy(attack_model.state_dict())
            temp_model.load_state_dict(attack_weights)
            temp_loader = DataLoader(dataset=dataset1, shuffle=False, batch_size=1, sampler=train_sampler)
            score_dict = score_datapoint_weights(args, temp_model, adversarial_model, device, temp_loader)

            sorted_scores = sorted(score_dict, key=score_dict.get) 
            attack_sampler = SequentialSampler(sorted_scores)

            attack_loader = DataLoader(dataset=dataset1, shuffle=False, batch_size=200, sampler=attack_sampler)

            train(args, attack_model, device, attack_loader, attack_optimizer, epoch)
            test(attack_model, device, test_loader)
            scheduler.step()
    else:
        print("else")
        attack_loader = DataLoader(dataset=dataset1, shuffle=False, batch_size=200, sampler=attack_sampler)
        scheduler = StepLR(attack_optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, adversarial_model, device, attack_loader, attack_optimizer, epoch)
            test(adversarial_model, device, test_loader)
            scheduler.step()


    if args.save_model:
        torch.save(attack_model.state_dict(), "mnist_cnn.pt")
    


if __name__ == '__main__':
    main()