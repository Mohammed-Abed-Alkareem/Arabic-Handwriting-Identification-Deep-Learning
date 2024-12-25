import torch
import torch.nn as nn
import torch.nn.functional as F

class NetCNN_1(nn.Module):
    """
    Baseline CNN:
    - 2 Conv Layers
    - Max Pooling with kernel_size=2, stride=2
    - Padding=1 in conv layers
    - Input: 1 x 224 x 224
    """
    def __init__(self, num_classes=82):
        super(NetCNN_1, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, 
                               kernel_size=3, stride=1, padding=1)  # Output: 16 x 224 x 224
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                               kernel_size=3, stride=1, padding=1)  # Output: 32 x 224 x 224
        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces to 32 x 112 x 112 after first pool
        # Fully Connected Layers
        # After two pools:
        # First Pool: 224 -> 112
        # Second Pool: 112 -> 56
        # So spatial size: 32 x 56 x 56
        self.fc1 = nn.Linear(32 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))  # (16, 112, 112)
        # Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # (32, 56, 56)
        # Flatten
        x = x.view(x.size(0), -1)             # (32*56*56) = 100352
        # FC1 -> ReLU
        x = F.relu(self.fc1(x))
        # FC2
        x = self.fc2(x)
        return x


class NetCNN_2(nn.Module):
    """
    Enhanced CNN:
    - 3 Conv Layers
    - Average Pooling with kernel_size=2, stride=2
    - Padding=0 in conv layers
    - Batch Normalization and Dropout
    - Input: 1 x 224 x 224
    """
    def __init__(self, num_classes=82, dropout_p=0.4):
        super(NetCNN_2, self).__init__()
        # Convolutional Layers with padding=0
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)  # Output: 16 x 222 x 222
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0) # Output: 32 x 220 x 220
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) # Output: 64 x 218 x 218
        self.bn3 = nn.BatchNorm2d(64)
        # Pooling Layer
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Reduces to 64 x 109 x 109
        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)
        # Fully Connected Layers
    
        self.fc1 = nn.Linear(64 * 26 * 26, 512)  # 64*109*109=753,664
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv1 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (16, 222, 222) -> Pool -> (16, 111, 111)
        # Conv2 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (32, 109, 109) -> Pool -> (32, 54, 54)
        # Conv3 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (64, 52, 52) -> Pool -> (64, 26, 26)
        # Flatten
        x = x.view(x.size(0), -1)                        # (64*26*26) = 43,264
        # FC1 -> ReLU -> Dropout
        x = self.dropout(F.relu(self.fc1(x)))            # (512)
        # FC2
        x = self.fc2(x)                                   # (num_classes)
        return x



class NetCNN_3(nn.Module):
    """
    Enhanced CNN with Global Average Pooling:
    - 3 Conv Layers
    - Average Pooling with kernel_size=2, stride=2
    - Batch Normalization and Dropout
    - Global Average Pooling replaces FC layers
    - Input: 1 x 224 x 224
    """
    def __init__(self, num_classes=82, dropout_p=0.4):
        super(NetCNN_3, self).__init__()
        # Convolutional Layers with padding=0
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)  # Output: 16 x 222 x 222
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0) # Output: 32 x 220 x 220
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) # Output: 64 x 218 x 218
        self.bn3 = nn.BatchNorm2d(64)
        # Pooling Layer
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Reduces to 64 x 109 x 109
        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1,1))  # Output: 64 x 1 x 1
        # Fully Connected Layer
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Conv1 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (16, 222, 222) -> Pool -> (16, 111, 111)
        # Conv2 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (32, 109, 109) -> Pool -> (32, 54, 54)
        # Conv3 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (64, 52, 52) -> Pool -> (64, 26, 26)
        # Global Average Pooling
        x = self.gap(x)                                  # (64, 1, 1)
        # Flatten
        x = x.view(x.size(0), -1)                        # (64)
        # Dropout
        x = self.dropout(x)
        # FC
        x = self.fc(x)                                   # (num_classes)
        return x


class NetCNN_4(nn.Module):
    """
    Strided CNN with Global Average Pooling:
    - 3 Conv Layers with stride=2 to downsample
    - Batch Normalization
    - Global Average Pooling
    - Input: 1 x 224 x 224
    """
    def __init__(self, num_classes=82):
        super(NetCNN_4, self).__init__()
        # Convolutional Layers with stride=2
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # Output: 32 x 112 x 112
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # Output: 64 x 56 x 56
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)# Output: 128 x 28 x 28
        self.bn3 = nn.BatchNorm2d(128)
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1,1))  # Output: 128 x 1 x 1
        # Fully Connected Layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv1 -> BN -> ReLU
        x = F.relu(self.bn1(self.conv1(x)))  # (32, 112, 112)
        # Conv2 -> BN -> ReLU
        x = F.relu(self.bn2(self.conv2(x)))  # (64, 56, 56)
        # Conv3 -> BN -> ReLU
        x = F.relu(self.bn3(self.conv3(x)))  # (128, 28, 28)
        # Global Average Pooling
        x = self.gap(x)                        # (128, 1, 1)
        # Flatten
        x = x.view(x.size(0), -1)              # (128)
        # FC
        x = self.fc(x)                          # (num_classes)
        return x



class NetCNN_5(nn.Module):
    """
    Mixed Pooling CNN:
    - 4 Conv Layers
    - Alternates between MaxPool and AvgPool with varying kernel sizes and strides
    - Padding=1 in conv layers
    - Input: 1 x 224 x 224
    """
    def __init__(self, num_classes=82):
        super(NetCNN_5, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)   # Output: 16 x 224 x 224
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Output: 32 x 224 x 224
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64 x 224 x 224
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # Output: 128 x 224 x 224
        self.bn4 = nn.BatchNorm2d(128)
        # Pooling Layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)    # Reduces to 112x112
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)  # Maintains size
        # Fully Connected Layers
        # After layers:
        # Conv1 -> Pool1: 16 x 112 x 112
        # Conv2 -> Pool2: 32 x 112 x 112
        # Conv3 -> Pool1: 64 x 56 x 56
        # Conv4 -> Pool2: 128 x 56 x 56
        self.fc1 = nn.Linear(128 * 56 * 56, 512)  # 128*56*56=401,408
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv1 -> BN -> ReLU -> MaxPool
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (16, 112, 112)
        # Conv2 -> BN -> ReLU -> AvgPool
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (32, 112, 112)
        # Conv3 -> BN -> ReLU -> MaxPool
        x = self.pool1(F.relu(self.bn3(self.conv3(x))))  # (64, 56, 56)
        # Conv4 -> BN -> ReLU -> AvgPool
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))  # (128, 56, 56)
        # Flatten
        x = x.view(x.size(0), -1)                        # (128*56*56) = 401,408
        # FC1 -> ReLU
        x = F.relu(self.fc1(x))                           # (512)
        # FC2
        x = self.fc2(x)                                   # (num_classes)
        return x




class NetCNN_6(nn.Module):
    """
    Deeper CNN with Strided Convolutions and Average Pooling:
    - 5 Conv Layers
    - Average Pooling with kernel_size=2, stride=2
    - Stride=2 in some conv layers for downsampling
    - Padding=0 in conv layers
    - Batch Normalization and Dropout
    - Input: 1 x 224 x 224
    """
    def __init__(self, num_classes=82, dropout_p=0.5):
        super(NetCNN_6, self).__init__()
        # Convolutional Layers with stride=2 where applicable
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)   # Output: 16 x 222 x 222
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)  # Output: 32 x 110 x 110
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)  # Output: 64 x 108 x 108
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0) # Output: 128 x 53 x 53
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)# Output: 256 x 51 x 51
        self.bn5 = nn.BatchNorm2d(256)
        # Pooling Layer
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)                 # Reduces to 256 x 25 x 25
        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)
        # Fully Connected Layers
        # After pooling: 256 x 25 x 25
        self.fc1 = nn.Linear(256 * 5 * 5, 1024)  # 256*25*25=160,000
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Conv1 -> BN -> ReLU
        x = F.relu(self.bn1(self.conv1(x)))      # (16, 222, 222)
        # Conv2 -> BN -> ReLU -> AvgPool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (32, 110, 110) -> Pool -> (32, 55, 55)
        # Conv3 -> BN -> ReLU
        x = F.relu(self.bn3(self.conv3(x)))      # (64, 53, 53)
        # Conv4 -> BN -> ReLU -> AvgPool
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # (128, 53, 53) -> Pool -> (128, 26, 26)
        # Conv5 -> BN -> ReLU
        x = F.relu(self.bn5(self.conv5(x)))      # (256, 24, 24)
        # Pooling
        x = self.pool(x)                          # (256, 12, 12)
        # Flatten
        x = x.view(x.size(0), -1)                 # (256*12*12) = 36,864
        # FC1 -> ReLU -> Dropout
        x = self.dropout(F.relu(self.fc1(x)))     # (1024)
        # FC2
        x = self.fc2(x)                           # (num_classes)
        return x
