import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm




input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
lr = 0.001
batch_size = 64
num_epochs = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
    

train_loader = DataLoader(
    MNIST(root = '../', train = True, download = False, transform = transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    MNIST(root = '../', train = False, download = False, transform = transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=False
)

model = SimpleRNN(input_size, hidden_size, num_layers, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)




def check_acc(loader, model):
    assert loader.dataset.test

    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
if __name__ == '__main__':
    
    #train
    for epoch in tqdm(range(num_epochs)):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device = device).squeeze(1)
            targets = targets.to(device)
            
            scores = model(data)
            print(data.shape)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

    print(check_acc(test_loader, model))