import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN models


class NetCNN_1(nn.Module):

    '''
    CNN model with 2 convolutional layers and 2 fully connected layers
    '''
    def __init__(self):
        super(NetCNN_1, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)  # 3 input channels, 16 output channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling with 2x2 kernel
        self.conv2 = nn.Conv2d(6, 9, kernel_size=3, padding=1)  # 16 input channels, 32 output channels

        self.fc1 = nn.Linear(9 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 82)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x