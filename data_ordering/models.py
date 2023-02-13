import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class MNISTNet(nn.Module):
    def __init__(self, image_size):
        super(MNISTNet, self).__init__()
        # image_size = 28*28
        num_classes=10
        self.model_arch = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(image_size, 392), nn.ReLU(),
            nn.Linear(392, 49), nn.ReLU(), 
            nn.Linear(49, 49), nn.ReLU(), 
            nn.Linear(49, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        return self.model_arch(x)

# TODO: Setup model design
class CIFARNet(nn.Module):
    def __init__(self, image_size):
        super(CIFARNet, self).__init__()
        num_classes=10
        self.model_arch = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(image_size, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        return self.model_arch(x)