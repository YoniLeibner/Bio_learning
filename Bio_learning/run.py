import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from Bio_learning.dataloaders import get_data
from Bio_learning.model import SelectiveGroupModel
# import os, pickle
# import numpy as np
criterion = nn.L1Loss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10
loaders = get_data(data_name='cifar10')
model = SelectiveGroupModel(input_dim=32, num_groups=20, group_size=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in loaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Flatten the input images
        inputs_flat = inputs.view(inputs.size(0), -1, inputs.size(-1)).transpose(1,2)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, corr_reg, scores = model(inputs_flat)

        # Compute loss
        loss = criterion(outputs, labels) - 0.1 * corr_reg  # Adjust the weight of corr_reg as needed

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(loaders['train'])}")