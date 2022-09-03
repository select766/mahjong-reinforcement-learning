"""
教師あり学習
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .sv_dataset import MjxSVDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dst_dir")
    parser.add_argument("train")
    parser.add_argument("val")
    args = parser.parse_args()

    batch_size = 64
    trainset = MjxSVDataset(args.train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = MjxSVDataset(args.val)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    model = nn.Sequential(
        nn.Linear(16 * 34, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 181)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}, accuracy: {correct / total}')
                running_loss = 0.0
                correct = 0
                total = 0

        with torch.no_grad():
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # forward + backward + optimize
                outputs = model(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)

                # print statistics
                running_loss += loss.item()
            print(f'[{epoch + 1}, val] loss: {running_loss / len(testloader):.3f}, accuracy: {correct / total}')
    torch.save(model.state_dict(), dst_dir / "policy.pt")

if __name__ == "__main__":
    main()
