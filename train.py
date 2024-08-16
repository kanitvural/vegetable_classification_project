import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_dataset import VegetableDataset
from model import VegetableModel

# Paths
train_folder = "./data/train"
valid_folder = "./data/validation"
test_folder = "./data/test"

# Configs
num_classes = 15
batch_size = 8
num_epochs = 8
train_losses, val_losses = [], []


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(), # Converts values in the range [0, 255] to the range [0, 1]
    ]
)

# Dataset
train_dataset = VegetableDataset(train_folder, transform=transform)
val_dataset = VegetableDataset(valid_folder, transform=transform)
test_dataset = VegetableDataset(test_folder, transform=transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model
model = VegetableModel(num_classes=num_classes)
model.to(device) 

# Parameters
criterion = nn.CrossEntropyLoss() # Loss function for softmax
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Training
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0 

    for images, labels in tqdm(train_loader, desc="Training loop"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() 
        outputs = model(images) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step() 
        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation loop"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)

    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    # Results for each epoch
    print(
        f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}"
    )

    # Save Model
    os.makedirs("model", exist_ok=True)
    torch.save(obj=model.state_dict(), f=f"model/vegetable_{epoch}.pth")
