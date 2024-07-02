import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import dataloader
from torchvision import transforms
from tqdm import tqdm

class FeedForwardNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_classes):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_classes)

    def forward(self, x):
        return self.fc2(self.fc1(x))
    

in_channels = 28*28
hidden_channels = 200
out_classes = 10
lr = 3e-4
no_epochs = 5
batch_size = 32

train_loader = dataloader.DataLoader(
    datasets.MNIST(root = './', train = True, transform=transforms.ToTensor(), download = False),
    batch_size = 32,
    shuffle = True
)
test_loader = dataloader.DataLoader(
    datasets.MNIST(root = './', train = False, transform=transforms.ToTensor(), download = False),
    batch_size = 1,
    shuffle = False
)

model = FeedForwardNet(in_channels, hidden_channels, out_classes)
optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
loss_fn = nn.CrossEntropyLoss()

#training loop
for ep in tqdm(range(no_epochs)):
    for idx, (data, target) in enumerate(train_loader):
        data = data.view(data.shape[0], -1)
        out = model(data)
        print(target.shape)
        
        loss = loss_fn(target, out)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss)

#test loop
data_list = []
with torch.no_grad():
    for i in range(10):
        for idx, (data, target) in enumerate(train_loader):
            target.squeeze(1)
            if target[0] == i:
                data_list.append(model(data))
                continue
print(data_list)
    
        