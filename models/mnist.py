# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining - IOL Lab @ ZIB
# File:         models/mnist.py
# Description:  MNIST Models
# ===========================================================================
import torch


class Simple(torch.nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.fc1 = torch.nn.Linear(784, 512, bias=True)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(512, 10, bias=True)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class SimpleCNN(torch.nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer = torch.nn.Sequential(
            # Conv Layer block 1
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.25)
        )

        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(9216, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        """Perform forward."""
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
