import torch 
import torch.nn as nn
from enum import Enum

#---------------------------------------------------------------------#
class GNet_MLP(nn.Module):

    def __init__(self):
        super(GNet_MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=988, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(in_features=64, out_features=32)
        self.fc6 = nn.Linear(in_features=32, out_features=2)
        self.softmax = nn.Softmax(dim=-1) # For binary classification
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.softmax(x)
        return x
#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
class ModelType(Enum):
    last_epoch = 'last net'
    best_epoch = 'best net'
    base = 'base'
#---------------------------------------------------------------------#