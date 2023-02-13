
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models import *
from torch.optim.lr_scheduler import StepLR
from CustomSampler import CustomSampler


class testModelRunner():
    def __init__(self, args, image_size, train_dataset, test_dataset, candidate_list, test_kwargs):
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda")
        if args.model == "MNISTNet":
            self.model = MNISTNet(image_size).to(self.device)
        elif args.model == "CIFAR10":
            self.temp_model = CIFARNet(image_size).to(self.device)
        
        self.sampler = CustomSampler(candidate_list, len(train_dataset)-1)
        self.train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.batch_size, sampler=self.sampler)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=args.lr)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        self.scheduler = StepLR(self.optimizer, step_size=0.1, gamma=args.gamma)

    def train(self, device, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))

    def test(self, device):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
