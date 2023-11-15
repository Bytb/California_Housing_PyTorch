# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#       - forward pass: compute prediction
#       - backward pass: gradients
#       - update weights
import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import math


# ------PYTORCH CLASSES------#


# creating class dataset
class CustomDataset(Dataset):
    def __init__(self, dataset) -> None:
        # data loading
        # loading in dataset
        xy = dataset
        # splitting the values into the dependent and independent variables
        # NOTE: look up the notation for SPLICING arrays
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]).view(xy.shape[0], 1)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


# creating the network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


# -----MODEL SETUP-----#


# setup GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper parameters
input_size = 71
hidden_size = 64
num_classes = 1
num_epochs = 50
batch_size = 20
learning_rate = 0.0001

train_data = np.loadtxt(
    "/Users/calebfernandes/Desktop/CaliHousingPredictions_PyTorch/data/tf_train_df.csv",
    delimiter=",",
    dtype=np.float32,
    skiprows=1,
)
test_data = np.loadtxt(
    "/Users/calebfernandes/Desktop/CaliHousingPredictions_PyTorch/data/tf_test_df.csv",
    delimiter=",",
    dtype=np.float32,
    skiprows=1,
)
# turning dataset into a dataloader to be iterated over
# NOTE: SHAPE OF DATA: [20640, 73]
train = CustomDataset(train_data)
test = CustomDataset(test_data)
train = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

# setting up model, loss, and optimizer
model = NeuralNet(input_size, hidden_size)
MSE = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

# -----TRAINING LOOP-----
# NOTE: ASK JUSTIN ABOUT ITERATIONS AND BATCH_SIZE!!!
iterations = math.ceil(train.__len__() / batch_size)
print(train.__len__())
print()

for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train):
        inputs.reshape(-1, 71).to(device)
        labels = labels.to(device)

        # loss function and optimizer
        outputs = model(inputs)
        loss = MSE(outputs, labels)

        # backwards
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 100 == 0:
            print(
                f"epoch {epoch+1}/{num_epochs}, step {i+1}/{iterations}, loss = {loss.item():.4f}"
            )

# -----TESTING AND ACCURACY-----
with torch.no_grad():
    model.eval()
    n_correct = 0
    n_samples = 0
    for images, labels in test:
        # reshaping again
        images = images.reshape(-1, 71).to(device)
        labels = labels.to(device)
        # getting predicted values
        outputs = model(images)

        # predictions
        predictions = outputs
        n_samples += labels.shape[0]
        n_correct += abs(predictions - labels).sum().item()
        print(predictions, labels)

    acc = n_correct / n_samples
    print(f"Accuracy: {acc}")
