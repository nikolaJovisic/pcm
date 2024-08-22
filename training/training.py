import torch
from model.mammo_unet import get_mammo_unet
from dataset.dataloaders import get_dataloaders

import torch.nn as nn
import torch.optim as optim

def train_model():
    model = get_mammo_unet()
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader, val_loader, _ = get_dataloaders(batch_size=4)

    for epoch in range(10):
        model.train()
        train_loss = 0.0
        val_loss = 0.0

        for x, labels, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        with torch.no_grad():
            for x, labels, _ in val_loader:
                outputs = model(x)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

if __name__ == "__main__":
    train_model()