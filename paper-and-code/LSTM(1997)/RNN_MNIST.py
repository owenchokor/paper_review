import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST




input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
lr = 0.001
batch_size = 64
num_epochs = 2


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size_size)

        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
    

train_data = DataLoader(
    MNIST(root = './', train = True, download = True, transform = transforms.ToTensor()),
    batch_size=80,
    shuffle=True
)
test_data = DataLoader(
    MNIST(root = './', train = False, download = True, transform = transforms.ToTensor()),
    batch_size=80,
    shuffle=False
)

model = SimpleRNN(input_size, hidden_size, num_layers, num_classes)
