import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

class myVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        self.input2hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden2latent = nn.Linear(hidden_dim, z_dim)
        self.latent2hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden2output = nn.Linear(hidden_dim, z_dim)
        
    def encoder(self, x):
        x = self.input2hidden(x)
        mu = self.hidden2latent(x)
        sigma = self.hidden2latent(x)
        return mu, sigma

    def decoder(self, z):
        hid = self.latent2hidden(z)
        return self.hidden2output(hid)
    
    def forward(self,x):
        mu, sigma = self.encoder(x)
        eps = torch.rand_like(sigma)
        z = mu + sigma*eps
        return self.decoder(z), mu, sigma
    
input_dim = 28*28
hidden_dim = 200
z_dim = 20
batch_size = 16
lr = 3e-4
n_epochs = 3


model = myVAE(input_dim, hidden_dim, z_dim)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

train_loader = DataLoader(
    datasets.MNIST('./', train = True, transform=transforms.ToTensor(), download=False),
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    datasets.MNIST('./', train = False, transform=False, download=False),
    batch_size=1,
    shuffle=False
)

for ep in range(n_epochs):
    loop = tqdm(train_loader)
    for idx, (data, _) in enumerate(loop):
        data = data.view(data.shape[0], -1)
        out, mu, sigma = model(data)
        recon_loss = nn.MSELoss(data, out)
        kld = -torch.sum(1 + torch.log(torch.pow(sigma, 2)) - torch.pow(mu, 2) - torch.pow(sigma, 2))
        loss = recon_loss + kld
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss = loss.item())

