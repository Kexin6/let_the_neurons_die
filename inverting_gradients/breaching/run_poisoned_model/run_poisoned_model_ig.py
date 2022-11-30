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
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def train(args, model, device, train_loader, optimizer, epoch, run_type):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('----------', run_type)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, run_type):
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

    print('----------', run_type)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (100. * correct / len(test_loader.dataset))

class MNISTDataset(Dataset):
    def __init__(self, data, label, transform=None):
        # re-scaling to [0,1] by diving by 255.
        data = data.astype(np.float32)
        label = label.astype(np.int)
        self.data, self.label = data / 255., label
        self.transform = transform
        
    def __getitem__(self, i):
        sample_data, sample_label = self.data[i], int(self.label[i])
        sample_data = sample_data.reshape(28, 28)
        if self.transform:
            sample_data = self.transform(sample_data)
        return sample_data, sample_label
    
    def __len__(self):
        return len(self.data)


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
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
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
    device = torch.device("cpu")

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

    image_size = 28*28
    batch_size = 32

    mnist = loadmat("./mnist-original")
    mnist_data = mnist["data"].T  # [70,000, 784]
    mnist_label = mnist["label"][0]  # [70,000]

    # print(type(mnist_data))
    # print(mnist_data.shape)

    train_data = mnist_data[:60000]
    train_label = mnist_label[:60000]
    test_data = mnist_data[60000:]
    test_label = mnist_label[60000:]

    print(train_label)
    print(len(train_label))
    
    # exit()

    # train_ds = MNISTDataset(train_data, train_label, transform=transform)
    # test_ds = MNISTDataset(test_data, test_label, transform=transform)
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # model = custom_model.DenseNet(image_size).to(device)
    # attack_optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # scheduler = StepLR(attack_optimizer, step_size=10, gamma=args.gamma)
    # run_type = 'Original'
    test_original_acc = [93.91, 94.79, 95.85, 96.09, 96.3, 96.88, 96.78, 96.66]
    test_poisoned_acc = []
    poison_attack_weights = []
    # for epoch in range(0, args.epochs + 1):
    #     train(args, model, device, train_loader, attack_optimizer, epoch, run_type)
    #     acc = test(model, device, test_loader, run_type)
    #     scheduler.step()
    #     test_original_acc.append(acc)

    if args.attack_type == 'single':
        num_chosen = 1
        single_mnist = np.load(f'../from_data_ordering/num_{num_chosen}_reconstructed_user_data_11_single.npy')
        # print(single_mnist)
        # print(train_data[0])
        # print(len(single_mnist[0][0]))
        # single_mnist = np.reshape(single_mnist, (1, 784))
        # print(single_mnist)
        # print(len(single_mnist))
        single_mnist = np.reshape(single_mnist, (1, 784))
        single_mnist = single_mnist * 255

        print(np.min(single_mnist))
        print(np.max(single_mnist))

        # img = Image.fromarray(single_mnist[0])
        # img.show() # Show the image

        # exit()

        train_data_poisoned = np.concatenate((train_data, single_mnist), axis=0)
        print(len(train_data_poisoned))

        train_label_poisoned = np.concatenate((train_label, [num_chosen]), axis=0)
        print(len(train_label_poisoned))
        print(train_label_poisoned)

    if args.attack_type == 'single_all_num':
        
        train_data_poisoned = train_data
        train_label_poisoned = train_label
        for i in range(10):
            single_mnist = np.load(f'../from_data_ordering/num_{i}_reconstructed_user_data_11_single.npy')
            single_mnist = np.reshape(single_mnist, (1, 784))
            single_mnist = single_mnist * 255

            print(np.min(single_mnist))
            print(np.max(single_mnist))

            train_data_poisoned = np.concatenate((train_data_poisoned, single_mnist), axis=0)
            print(len(train_data_poisoned))

            train_label_poisoned = np.concatenate((train_label_poisoned, [i]), axis=0)
            print(len(train_label_poisoned))
            print(train_label_poisoned)

        

    if args.attack_type == 'batch':
        num_chosen = 1
        batch_mnist = np.load(f'../from_data_ordering/num_{num_chosen}_reconstructed_user_data_12_batch.npy')
        # print(single_mnist)
        # print(len(single_mnist[0][0]))
        num_poison = len(batch_mnist)
        batch_mnist = np.reshape(batch_mnist, (num_poison, 784))
        # print(single_mnist)
        print(len(batch_mnist))

        train_data_poisoned = np.concatenate((train_data, batch_mnist), axis=0)
        print(len(train_data_poisoned))

        train_label_poisoned = np.concatenate((train_label, np.full(shape=num_poison, fill_value=num_chosen)), axis=0)
        print(len(train_label_poisoned))
        print(train_label_poisoned)

    if args.attack_type == 'batch_all_num':

        train_data_poisoned = train_data
        train_label_poisoned = train_label
        for i in range(10):
            batch_mnist = np.load(f'../from_data_ordering/num_{i}_reconstructed_user_data_12_batch.npy')
            num_poison = len(batch_mnist)
            batch_mnist = np.reshape(batch_mnist, (num_poison, 784))
            # print(single_mnist)
            print(len(batch_mnist))

            train_data_poisoned = np.concatenate((train_data_poisoned, batch_mnist), axis=0)
            print(len(train_data_poisoned))

            train_label_poisoned = np.concatenate((train_label_poisoned, np.full(shape=num_poison, fill_value=i)), axis=0)
            print(len(train_label_poisoned))
            print(train_label_poisoned)
    
    train_ds = MNISTDataset(train_data_poisoned, train_label_poisoned, transform=transform)
    test_ds = MNISTDataset(test_data, test_label, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    poisoned_model = custom_model.DenseNet(image_size).to(device)
    attack_optimizer = optim.Adadelta(poisoned_model.parameters(), lr=args.lr)

    scheduler = StepLR(attack_optimizer, step_size=10, gamma=args.gamma)
    run_type = args.attack_type + ' poisoned'
    for epoch in range(0, args.epochs + 1):
        train(args, poisoned_model, device, train_loader, attack_optimizer, epoch, run_type)
        acc = test(poisoned_model, device, test_loader, run_type)
        scheduler.step()
        test_poisoned_acc.append(acc)

        attack_weights = poisoned_model.state_dict()['model_arch.5.weight'].data.cpu().detach().numpy()
        poison_attack_weights.append(np.sum(attack_weights))
    
    print(f'============{args.attack_type}============')
    print('Original Test Accuracy')
    print(test_original_acc)
    print()
    print(args.attack_type + ' poisoned test accuracy')
    print(test_poisoned_acc)
    print()
    print(args.attack_type + ' poisoned attack weights')
    print(poison_attack_weights)



if __name__ == '__main__':
    main()