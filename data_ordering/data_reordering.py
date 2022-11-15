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
from CustomSampler import CustomSampler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import custom_model
import adversary
import run_model


min_index = 0

# Algorithm
# Run the temp_model and adversarial_model on the datapoint
# Calculate their updates
# Compare their weight deltas (do we need to do a baseline compare or is gradient enough)
# Calculate the score based on the gradient
def score_datapoint_weights(args, temp_model, adversarial_model, device, train_loader, used_points):
    global min_index
    # base_model = copy.deepcopy(temp_model.state_dict())
    weights = []
    score_list = {}

    for param in temp_model.parameters():
        weights.append(param.clone())

    adversarial_weights = np.empty([0]) # weights of adversary
    for layer in adversarial_model.state_dict():
        if 'weight' in layer:
            adversarial_weights = np.concatenate((adversarial_weights, adversarial_model.state_dict()[layer].data.cpu().detach().numpy()), axis=None)
    
    min_index = np.argmin(np.abs(adversarial_weights))
    min_adversary_weight = adversarial_weights[min_index]
    print('Min Adversarial weight ' + str(min_adversary_weight))
    
    # for param in adversarial_model.parameters():
    #     adversarial_weights = np.concatenate((adversarial_weights, param.cpu().detach().numpy()), axis=None)
    # print(np.sum(np.abs(adversarial_weights)))
    # print(adversarial_weights)
    i = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Set model
        if not i in used_points:
            if i % 1000 == 0:
                print("On data point: " + str(i))


            
            # for module in temp_model.model_arch.children():
            #     if isinstance(module, nn.Linear):
            #         module.reset_parameters()
            
            #temp_model.load_state_dict(base_model)
        
            
            # for param in temp_model.parameters():
            #     temp_weights = np.concatenate((temp_weights, param.cpu().detach().numpy()), axis=None)
            # print(np.sum(np.abs(temp_weights)))
            temp_model.train()
            temp_optimizer = optim.Adadelta(temp_model.parameters(), lr=args.lr)
            data, target = data.to(device), target.to(device)
            
            # Temp Model
            temp_optimizer.zero_grad()
            output_temp = temp_model(data)
            loss_temp = F.nll_loss(output_temp, target)
            loss_temp.backward()
            temp_optimizer.step()

            # Dynamic single weight
            temp_weights = np.empty([0]) # weights after backprop
            for layer in temp_model.state_dict():
                if 'weight' in layer: 
                    temp_weights = np.concatenate((temp_weights, temp_model.state_dict()[layer].data.cpu().detach().numpy()), axis=None)
            # score_np = abs(min_adversary_weight - temp_weights[min_index])
            score_np = abs(temp_weights[min_index])
            score_list[i] = score_np


            # if (args.score_heuristic == 0):
            #     temp_weights = np.empty([0]) # weights after backprop
            #     for layer in temp_model.state_dict():
            #         if 'weight' in layer: 
            #             temp_weights = np.concatenate((temp_weights, temp_model.state_dict()[layer].data.cpu().detach().numpy()), axis=None)

                
            #     gradient_delta = np.abs(np.subtract(temp_weights, adversarial_weights))
            #     score_np = np.sum(gradient_delta)
            #     score_list[i] = score_np
            # elif (args.score_heuristic == 1):
            #     temp_weights = np.empty([0]) # weights after backprop
            #     scaling_lambda = np.empty([0]) # taking each weight into consideration
            #     for layer in temp_model.state_dict():
            #         if 'weight' in layer:
            #             temp_np = temp_model.state_dict()[layer].data.cpu().detach().numpy()
            #             w_max = np.amax(temp_np)
            #             temp_np = 1 - temp_np / w_max
            #             scaling_lambda = np.concatenate((scaling_lambda, temp_np), axis=None)
            #             temp_weights = np.concatenate((temp_weights, temp_model.state_dict()[layer].data.cpu().detach().numpy()), axis=None)
                        
            #     score_np = abs(min_adversary_weight - temp_weights[min_index])
            #     #gradient_delta = np.abs(np.subtract(temp_weights, adversarial_weights))
            #     #score_np = np.sum(np.multiply(np.transpose(scaling_lambda), np.sum(gradient_delta)))

            #     score_list[i] = score_np
        i += 1
            
    return score_list

def get_attack_loader(args, temp_model, adversarial_model, device, temp_loader, dataset1, used_points):
    score_dict = score_datapoint_weights(args, temp_model, adversarial_model, device, temp_loader, used_points)
    # Higher the score the worse it is
    sorted_scores = sorted(score_dict, key=score_dict.get) 
    
    attack_sampler = SequentialSampler(sorted_scores)
    attack_loader = DataLoader(dataset=dataset1, shuffle=False, batch_size=1, sampler=attack_sampler)
    return attack_loader

def get_dynamic_attack_loader(args, temp_model, adversarial_model, device, temp_loader, dataset1, used_points):
    score_dict = score_datapoint_weights(args, temp_model, adversarial_model, device, temp_loader, used_points)
    # Higher the score the worse it is
    
    sorted_scores = sorted(score_dict, key=score_dict.get) 
    used_points = []
    if len(sorted_scores) > 0:
        used_points.append(sorted_scores[0])

    print('Elected Point: ' + str(sorted_scores[0]))
    print('Score ' + str(score_dict.get(sorted_scores[0])))
    
    # full_points = copy.deepcopy(used_points)
    # full_points.append(sorted_scores)
    sampler = []
    sampler.append(sorted_scores[0])
    attack_sampler = SequentialSampler(sampler)
    
    attack_loader = DataLoader(dataset=dataset1, shuffle=False, batch_size=1, sampler=attack_sampler)
    return attack_loader, used_points

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


def train_dynamic_attack(args, model, device, optimizer, epoch, temp_model, adversarial_model, temp_loader, dataset1):
    global min_index
    model.train()
    used_points = []

    # Steps. 1 Get the attack loader
    # temp_model = copy.deepcopy(model)
    image_size = 32*32*3
    temp_model = custom_model.DenseNet(image_size).to(device)
    attack_loader, used_points = get_dynamic_attack_loader(args, temp_model, adversarial_model, device, temp_loader, dataset1, used_points)
    print(used_points)
    # Runs a single datapoint
    for batch_idx, (data, target) in enumerate(attack_loader):
         # Reset for next point
       
        data, target = data.to(device), target.to(device)

        # Trying resetting the optimizer? 
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(attack_loader.dataset),
            100. * batch_idx / len(attack_loader), loss.item()))

        attack_weights = np.empty([0]) # weights of adversary
        for layer in model.state_dict():
            if 'weight' in layer:
                attack_weights = np.concatenate((attack_weights, model.state_dict()[layer].data.cpu().detach().numpy()), axis=None)
        print("Knocked out weight")
        print(attack_weights[min_index])
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

def load_attacker_weight_list(weight_list, network):
   
    i = 0
    for layer in network.state_dict():
        print(layer)
        if 'weight' in layer: 
            new_weights = torch.from_numpy(weight_list[i])
            network.state_dict()[layer].data.copy_(new_weights)
            i += 1
    return network

# For ordering array of dictionaries by dictionary values
def sum_getter(arr):
    return sum(arr.values())

# For merging array of dictionaries to a single dictionary
def MergeDicts(arr):
    result = {}
    for d in arr:
        result.update(d)
    return result

def main():
    # Training settings
    global min_index
    parser = argparse.ArgumentParser(description='Data Ordering Attack', formatter_class=argparse.RawTextHelpFormatter)
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
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--attack-type', type=str, default='single',
                        help='Pick attack type')
    parser.add_argument('--score-heuristic', type=int, default=1,
                        help=textwrap.dedent('''
                        Different score heuristics to try 
                        0: only consider distance 
                        1: distance + original weight
                        '''))
    parser.add_argument('--attack-batch-size', type=int, default=1000,
                        help='Number of datapoints in each batch for batch ordering attack')
    parser.add_argument('--dataset', type=str, default='single',
                        help='Pick Dataset: MNIST, CIFAR10')
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

    if args.dataset == 'MNIST':
        dataset1 = datasets.MNIST('../data', train=False, download=True,
                        transform=transform)
        dataset2 = datasets.MNIST('../data', train=True,
                        transform=transform)
        image_size = 28*28
    elif args.dataset == 'CIFAR10':
        dataset1 = datasets.CIFAR10('../data', train=False, download=True,
                        transform=transform)
        dataset2 = datasets.CIFAR10('../data', train=True,
                        transform=transform)
        image_size=32*32*3

    # In order dataset
    dataset_size = len(dataset1)
    print(dataset_size)
    dataset_indices = list(range(dataset_size))
    train_sampler = SequentialSampler(dataset_indices)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
   

    # Define Models

    adversarial_model = custom_model.DenseNet(image_size).to(device)
    temp_model = custom_model.DenseNet(image_size).to(device)
    attack_model = custom_model.DenseNet(image_size).to(device)

    
    ### Getting knock-out weights directly 
    adversarial_weights = [] # weights of adversary    
    for layer in adversarial_model.state_dict():
        if 'weight' in layer: 
            adversarial_weights.append(adversarial_model.state_dict()[layer].data.cpu().detach().numpy())
    
    adversarial_weights = adversary.get_target_weights(adversarial_weights, 0.4)
    adversarial_model = load_attacker_weight_list(adversarial_weights, adversarial_model)

    # Getting knock-out weights from file
    # weight_file = 'weights_target_array.npy'
    # adversarial_model = load_attacker(weight_file, adversarial_model)

    attack_optimizer = optim.Adadelta(attack_model.parameters(), lr=args.lr)
    # Create a sampler for the attack

    attack_sampler = SequentialSampler(dataset_indices)
    
    # Specific attacks
    if args.attack_type == 'single':
        # Run through all the datapoints once to get score and do it in one shot
        print("Running simple data order attack")
        temp_loader = DataLoader(dataset=dataset1, shuffle=False, batch_size=1, sampler=train_sampler)
        attack_loader = get_attack_loader(args, temp_model, adversarial_model, device, temp_loader, dataset1, [])
        scheduler = StepLR(attack_optimizer, step_size=1, gamma=args.gamma)

        for epoch in range(1, args.epochs + 1):
            train(args, attack_model, device, attack_loader, attack_optimizer, epoch)
            test(attack_model, device, test_loader)
            scheduler.step()
        
        attack_weights = np.empty([0]) # weights of adversary
        for layer in adversarial_model.state_dict():
            if 'weight' in layer:
                attack_weights = np.concatenate((attack_weights, attack_model.state_dict()[layer].data.cpu().detach().numpy()), axis=None)
        print(attack_weights[min_index])
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

            attack_loader = DataLoader(dataset=dataset1, shuffle=False, batch_size=1, sampler=attack_sampler)

            train(args, attack_model, device, attack_loader, attack_optimizer, epoch)
            test(attack_model, device, test_loader)
            scheduler.step()
    elif args.attack_type == 'dynamic':
         # Run through all the datapoints once to get score and do it in one shot
        print("Running dynamic data order attack")
        temp_loader = DataLoader(dataset=dataset1, shuffle=False, batch_size=1, sampler=train_sampler)
        
        scheduler = StepLR(attack_optimizer, step_size=1, gamma=args.gamma)

        for epoch in range(0, args.epochs + 1):
            train_dynamic_attack(args, attack_model, device, attack_optimizer, epoch, temp_model, adversarial_model, temp_loader, dataset1)
            test(attack_model, device, test_loader)
            scheduler.step()
        
        attack_weights = np.empty([0]) # weights of adversary
        for layer in attack_model.state_dict():
            if 'weight' in layer:
                attack_weights = np.concatenate((attack_weights, attack_model.state_dict()[layer].data.cpu().detach().numpy()), axis=None)
        print("Knocked out weight")
        print(attack_weights[min_index])
    elif args.attack_type == 'batch':
        # Run through all the datapoints once to get score and do it in one shot in batches
        print("Running simple batch data order attack")
        # TODO: Currently we ask users to input batch_size. 
        # This may not be intuitive, later on we may change user args input to num_batches
        # and calculate batch_size by modulo calculation
        temp_loader = DataLoader(dataset=dataset1, shuffle=False, batch_size=args.attack_batch_size, sampler=train_sampler)
        starting_idx = 0
        score_dicts_arr = []
    
        for _ in enumerate(temp_loader):
            print('For batch: ' + str(_[0]))
            new_scores = score_datapoint_weights(args, temp_model, adversarial_model, device, temp_loader, starting_idx)
            starting_idx += len(new_scores)
            score_dicts_arr.append(new_scores)
            # print(score_dicts_arr)
        print("Data Order for batch")
        # Higher the score the worse it is
        sorted_scores_dict_arr = sorted(score_dicts_arr, key=sum_getter) # currently sorted_scores_dict_arr is still an array of dictionaries
        print(sorted_scores_dict_arr)
        sorted_scores = MergeDicts(sorted_scores_dict_arr)
        print(sorted_scores) # should be a large sorted dictionary
        attack_sampler = SequentialSampler(sorted_scores)

        attack_loader = DataLoader(dataset=dataset1, shuffle=False, batch_size=100, sampler=attack_sampler)
        scheduler = StepLR(attack_optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(0, args.epochs + 1):
            train(args, adversarial_model, device, attack_loader, attack_optimizer, epoch)
            test(adversarial_model, device, test_loader)
            scheduler.step()
    elif args.attack_type == 'test':

        attack_weights = np.empty([0]) # weights of adversary
        for layer in attack_model.state_dict():
            if 'weight' in layer:
                attack_weights = np.concatenate((attack_weights, attack_model.state_dict()[layer].data.cpu().detach().numpy()), axis=None)
        min_index = np.argmin(np.abs(attack_weights))
        
        candidate_list = [1044, 217, 4571, 7942, 7380, 2250, 3030]
        
        # candidate_list = []
        temp_list = []

        for j in range(20):
            min_value = 1000
            candidate_index = 0
            for i in range (10000):
                temp_list = copy.deepcopy(candidate_list)
                # dataset_indices = list(range(i, i+1000))
                # if (i == 300):
                #     dataset_indices = list(range(i, i+5))
                if not i in candidate_list:
                    attack_sampler = CustomSampler(temp_list, i)
                    attack_loader = DataLoader(dataset=dataset1, shuffle=False, batch_size=10, sampler=attack_sampler)

                    test_1 = run_model.runModel(image_size)
                
                    value = copy.deepcopy(test_1.run_temp(attack_loader, min_index))
                    
                    if value < min_value:
                        candidate_index = i
                        min_value = value
            
            
            print('On run ' + str(j))
            print('Added_index ' + str(candidate_index))
            print('Current Weight ' + str(min_value))
            candidate_list.append(candidate_index)
        
        print('-------------------------------------------')
        print('Final weight ' + str(min_value))
        print('Final Candidate list' + str(candidate_list))

        



    else:
        print("else")
        attack_loader = DataLoader(dataset=dataset1, shuffle=False, batch_size=1, sampler=attack_sampler)
        scheduler = StepLR(attack_optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(0, args.epochs + 1):
            train(args, adversarial_model, device, attack_loader, attack_optimizer, epoch)
            test(adversarial_model, device, test_loader)
            scheduler.step()
            
            attack_weights = np.empty([0]) # weights of adversary
            for layer in attack_model.state_dict():
                if 'weight' in layer:
                    attack_weights = np.concatenate((attack_weights, attack_model.state_dict()[layer].data.cpu().detach().numpy()), axis=None)
            print("Knocked out weight")
            print(attack_weights[602155])

    if args.save_model:
        torch.save(attack_model.state_dict(), "cifar10_cnn.pt")
    


if __name__ == '__main__':
    main()