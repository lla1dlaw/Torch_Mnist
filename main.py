

#Feed Forward Neural Network Trained and Tested on MNIST dataset
import pprint
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# devicce config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available(): 
    print("Using GPU")
else:
    print('Using CPU')

# hyperparameters
input_size = 784  # 28x28
num_classes = 10
hidden_size = 100
num_hidden = 4
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')


# plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden, num_classes):
        super(NeuralNet, self).__init__()

        self.input_size = input_size
        
        self.layers = [nn.Linear(input_size, hidden_size)]

        for i in range(num_hidden-2): 
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        self.layers.append(nn.Linear(hidden_size, num_classes))
        self.layers = nn.ModuleList(self.layers)
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        # no activation and no softmax at the end
        return x


model = NeuralNet(input_size, hidden_size, num_hidden, num_classes).to(device)

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

#training loop
n_total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # (100, 1, 28, 28)
        # (100, 784)
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1)% 100  == 0:
            print(f'epoch {epoch+1}, step {i+1}/{n_total_step}, loss= {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

def save_csv(model, path):
    state = model.state_dict()

    os.makedirs(path, exist_ok=True)
    for key, value in state.items():
        pd.DataFrame(value.numpy()).to_csv(f"{path}\\{key}.csv", index=False, header=False)
    
    print(f"Model saved to: {os.getcwd()}\\{path}")


if input("See state dictionary of the model? (y/n): ").lower() == "y":
    print(f"\nState Dictionary Type: {type(model.state_dict())}")
    pprint.pp(model.state_dict())

if input("Would you like to save this model? (y/n): ").lower() == "y":
    save_csv(model, "params")

