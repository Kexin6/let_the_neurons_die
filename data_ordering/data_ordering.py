from __future__ import print_function
import copy, random, torch, argparse, csv
from torchvision import datasets, transforms
from CustomSampler import CustomSampler
from torch.utils.data import DataLoader
from attackModelRunner import attackModelRunner
from testModelRunner import testModelRunner
from data_utils import *

def main():
    # Arg Parsing
    parser = argparse.ArgumentParser(description='Data Ordering Attack', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--method', type=str, default='ddoa', help='Pick method type: ddoa, test, save')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Pick Dataset: MNIST, CIFAR10')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of datapoints in each batch for batch ordering attack')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--sample-size', type=int, default=5000, help='Number of datapoints to test during DDOA')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--epochs', type=int, default=7, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--layer', type=int, default=5, help='Pick NN layer number to attack')
    parser.add_argument('--datapoint-num', type=int, default=200, help='Number of data ordered attacks to run')
    parser.add_argument('--candidate-pt', type=str, default='mnist_model_candidate.pt', help='Name to save the model weights as')
    parser.add_argument('--model', type=str, default='MNISTNet', help='Base model selection from models.py')
    parser.add_argument('--csv-name', type=str, default='MNISTCandidateList.csv', help='CSV for candidate list')
    args = parser.parse_args()

    # Device Setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Dataset Selection
    if args.dataset == 'MNIST':
        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, transform=transform)
        image_size = 28*28
    elif args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10('../data', train=False, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('../data', train=True, transform=transform)
        image_size=32*32*3

    # In order dataset
    dataset_size = len(train_dataset)
    # Set the layer to attack
    attack_layer = 'model_arch.'+ str(args.layer) +'.weight'
    
    # Attack List
    if args.method == 'ddoa':
        # Just killing the first neuron
        print('Dynamic Data Ordering Attack')
        candidate_list = []

        sample = random.sample(range(0, dataset_size), args.sample_size)
        for j in range(args.datapoint_num):
            min_value = 10000
            candidate_index = 0
            for i in sample:
                if not i in candidate_list:
                    temp_list = copy.deepcopy(candidate_list)
                    attack_sampler = CustomSampler(temp_list, i)
                    attack_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=1, sampler=attack_sampler)
                    attack_runner = attackModelRunner(args, image_size)
                    value = attack_runner.run_attack(attack_loader, attack_layer)
                    
                    if value < min_value:
                        # Save new candidate index
                        candidate_index = i
                        min_value = value
            
            print('On run ' + str(j))
            print('Added_index ' + str(candidate_index))
            print('Current Score ' + str(min_value))
            candidate_list.append(candidate_index)

        save_candidates(args.csv_name, candidate_list)

    elif args.method == 'test':
        print("Testing Candidate List")
        candidate_list = load_candidates(args.csv_name)
        
        all_points = []
        for i in range (dataset_size):
            if not i in candidate_list:
                all_points.append(i)
        candidate_list = candidate_list + all_points
        test_runner = testModelRunner(args, image_size, train_dataset, test_dataset, candidate_list, test_kwargs)
        
        for epoch in range(0, args.epochs + 1):
            test_runner.train(device, epoch)
            test_runner.test(device)
            test_runner.scheduler.step()
            weights = test_runner.model.state_dict()[attack_layer].data.cpu().detach().numpy()
            if epoch == 0:
                prev_weights = weights
            else:
                total = 0
                for i in range(len(weights)):
                    total += abs(weights[i] - prev_weights[i])
                print('AVERAGE Weight Change')
                print (total/(49))
                prev_weights = copy.deepcopy(weights)
           
    else:
        print("Saving Current Model Progress")
        candidate_list = load_candidates(args.csv_name)
        save_model_candidates(args, image_size, device, train_dataset, candidate_list, attack_layer, args.candidate_pt)
         

if __name__ == '__main__':
    main()