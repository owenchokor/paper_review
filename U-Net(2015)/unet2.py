import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features = [64, 128, 256, 512]
    ):
        super(UNET, self).__init__()
        
        self.upconvs = nn.ModuleList()
        self.updoubles = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature*2, feature, 2, 2),
            )
        for feature in reversed(features):
            self.updoubles.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = DoubleConv(features[0], out_channels)


    def forward(self, x):
        skip_connections = []
        for md in self.downs:
            x = md(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections.reverse()

        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            try:
                x = torch.cat((skip_connections[i], x), dim = 1)
            except RuntimeError:
                x = TF.resize(x, size = skip_connections[i].shape[2:])
            x = self.updoubles[i](x)
        return self.final_conv(x)
        
def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()