# E configuration of ConvNet (VGG)
import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGG, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(512*7*7, 4096, bias=False)
        self.fc2 = nn.Linear(4096, 4096, bias=False)
        self.fc3 = nn.Linear(4096, out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.smax = nn.Softmax()
        

    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool(x)
        x = self.convblock2(x)
        x = self.pool(x)
        x = self.convblock3(x)
        x = self.pool(x)
        x = self.convblock4(x)
        x = self.pool(x)
        x = self.convblock5(x)
        x = self.pool(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.smax(x)
        return x

def test():
    test_tensor = torch.rand((1, 3, 224, 224))
    model = VGG(3, 1000)
    print(model(test_tensor).shape)

if __name__ == "__main__":
    test()