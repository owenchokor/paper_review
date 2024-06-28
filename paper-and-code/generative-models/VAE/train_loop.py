import torch
import torch.nn as nn
import torchvision.datasets as Datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import VAE
from tqdm import tqdm

train_size = 28*28
batch_size = 16
hidden_length = 200
z_length = 20
epochs = 3
lr = 3e-4

train_dataset = DataLoader(
    Datasets.MNIST('C:\\Users\\82103\\paper_review\\paper-and-code', train = True, transform=transforms.ToTensor(), download=False),
    batch_size=batch_size,
    shuffle=True
)
test_dataset = DataLoader(
    Datasets.MNIST('C:\\Users\\82103\\paper_review\\paper-and-code', train = False, transform=transforms.ToTensor(), download=False),
    batch_size=batch_size,
    shuffle=False
)

def inference(digit, num_example):
    images = []
    idx = 0
    for x, y in test_dataset:
        if y == idx:
            images.append(x)
            idx+=1
        if idx == 10:
            break
    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))
    mu, sigma = encodings_digit(digit)

model = VAE.VAE(train_size, hidden_length, z_length)
optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.BCELoss(reduction='sum')

for epoch in range(epochs):
    loop = tqdm(enumerate(train_dataset))
    for i, (img, _) in loop:
        img = img.view(img.shape[0], train_size)
        img_recon, mu, sigma = model(img)
        recon_loss = loss_func(img_recon, img)
        kld = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        loss = recon_loss + kld
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss = loss.item())
        

for idx in range(10):
    inference(idx, num_example=1)



