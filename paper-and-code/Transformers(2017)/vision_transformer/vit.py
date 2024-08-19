import torch
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, num_heads, num_layers):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(64, self.num_patches + 1, dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, num_heads),
            num_layers
        )
        self.classification_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((self.positional_embedding, x), dim=1)
        
        x = self.transformer_encoder(x)
        x = x[:, 0]  # Take the first token as the representation of the whole image
        x = self.classification_head(x)
        return F.log_softmax(x, dim=-1)

# Create an instance of the ViT model
model = ViT(image_size=28, patch_size=4, num_classes=10, dim=256, num_heads=8, num_layers=6)

import torchvision.transforms as transforms

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='C:\\Users\\82103\\paper_review\\paper-and-code', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='C:\\Users\\82103\\paper_review\\paper-and-code', train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
model.to(device)
loop = tqdm(range(10))

for epoch in loop:
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() 
        loop.set_description(f"Epoch [{epoch+1}/10]")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader)}")

# Test the model
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy}%")