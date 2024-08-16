"""
This trains a convolutional neural network from ResNet50 V2 weights with PyTorch.
There are 50 total layers (based on ResNet 50 architecture) and the final layer is modified to only have four classes.
This uses cross-entropy loss as its loss function and the Adam optimiser.
There should be four folders in the ./data folder as it expects 4 classes for classification.
Random transformations are performed in each epoch to randomly transform the dataset.
The learning rate remains constant, regardless of performance of the model across epochs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on GPU.") if torch.cuda.is_available() else print(f"Training on CPU :(")

# Paths and hyperparameters
data_dir = './data'
batch_size = 32
num_epochs = 5
learning_rate = 0.001
train_val_split = 0.75  # 75% training, 25% testing

# Define transformations with data augmentation for the training set
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
    transforms.RandomRotation(20),      # Randomly rotate the image by up to 20 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly change brightness, contrast, saturation, and hue
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pre-trained models expect this normalization
])

# Define transformations for the testing set (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset with the training transform initially
dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# Calculate sizes for train and test sets
train_size = int(train_val_split * len(dataset))
test_size = len(dataset) - train_size

# Split dataset into training and testing sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Apply the test transform to the test dataset
test_dataset.dataset.transform = test_transform

# Create DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load a pre-trained ResNet model and modify the final layer --> this should only happen the first time, then it should be cached.
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, 4)  # Adjust the final layer for 4 classes
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    print(f"Commencing epoch {epoch + 1} of {num_epochs}.")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%')

    # Validation step
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
    val_accuracy = 100 * correct / total
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

# Final evaluation on the test set
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

print(f'Final Test Accuracy: {100 * correct / total:.2f}%')

torch.save(model.state_dict(), 'resnet50_fundus_trained.pth')




# TO DO:
# Print more statistics and eval metrics.
# Save the model. DONE
# Load the model. INSTRUCTIONS DONE
# Streamline usage (command-line), load folders of images or individual images.
# Structure your project properly lol.

