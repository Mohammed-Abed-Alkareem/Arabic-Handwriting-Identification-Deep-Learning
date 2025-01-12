import torch
import torch.nn as nn
import torch.nn.functional as F


class NetCNN_0(nn.Module):
    def __init__(self, num_classes=82, input_size=(1, 224, 224)) -> None:
        super().__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=10, out_channels=40, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=40, out_channels=80, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=28 * 28 * 80, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes)
        )
    def forward(self, x):
        x = self.cnn_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
        return x



class NetCNN_1(nn.Module):
    """
    - 2 Conv Layers
     - 2 Conv Layers
    - Max Pooling with kernel_size=2, stride=2
    - Padding=1 in conv layers
    - Input: 1 x 224 x 224
    """
    def __init__(self, num_classes=82, input_size=(1, 224, 224)):
        super(NetCNN_1, self).__init__()
        
        # Convolutional Layers
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),  # 16 x 224 x 224
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                         # 16 x 112 x 112
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 32 x 112 x 112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                                          # 32 x 56 x 56
        )

        # Fully Connected Layers
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=32*56*56, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x):
        x = self.cnn_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
        return x


class NetCNN_2(nn.Module):
    """
     - 3 Conv Layers
    - Average Pooling with kernel_size=2, stride=2
    - Padding=0 in conv layers
    - Batch Normalization and Dropout
    - Input: 1 x 224 x 224
    """
    def __init__(self, num_classes=82, dropout_p=0.4, input_size=(1, 224, 224)):
        super(NetCNN_2, self).__init__()
        
        # Convolutional Layers
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0),  # 16 x 222 x 222
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),                                       # 16 x 111 x 111
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0), # 32 x 109 x 109
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),                                       # 32 x 54 x 54
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0), # 64 x 52 x 52
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)                                        # 64 x 26 x 26
        )
    
       # Fully Connected Layers
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=64*26*26, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes)
        )
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, x):
        x = self.cnn_layer(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc_layer(x)
        return x



class NetCNN_3(nn.Module):
    """
    - 3 Conv Layers
    - Average Pooling with kernel_size=2, stride=2
    - Batch Normalization and Dropout
    - Global Average Pooling replaces FC layers
    - Input: 1 x 224 x 224
    """
    def __init__(self, num_classes=82, dropout_p=0.4, input_size=(1, 224, 224)):
        super(NetCNN_3, self).__init__()
        
        # Convolutional Layers
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0),  # 16 x 222 x 222
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),                                       # 16 x 111 x 111
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0), # 32 x 109 x 109
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),                                       # 32 x 54 x 54
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0), # 64 x 52 x 52
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)                                        # 64 x 26 x 26
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 64 x 1 x 1
        
        # Fully Connected Layer
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=64, out_features=num_classes)
        )
        
    def forward(self, x):
        x = self.cnn_layer(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
        return x


class NetCNN_4(nn.Module):
    """
    - 3 Conv Layers with stride=2 to downsample
    - Batch Normalization
    - Global Average Pooling
    - Input: 1 x 224 x 224
    """
    def __init__(self, num_classes=82, input_size=(1, 224, 224)):
        super(NetCNN_4, self).__init__()
        
        # Convolutional Layers
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),  # 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 64 x 56 x 56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),# 128 x 28 x 28
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 128 x 1 x 1
        
        # Fully Connected Layer
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=128, out_features=num_classes)
        )
        
    def forward(self, x):
        x = self.cnn_layer(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
        return x




class NetCNN_5(nn.Module):
    """
    - 4 Conv Layers
    - Alternates between MaxPool and AvgPool with varying kernel sizes and strides
    - Padding=1 in conv layers
    - Batch Normalization
    - Input: 1 x 224 x 224
    """
    def __init__(self, num_classes=82, input_size=(1, 224, 224)):
        super(NetCNN_5, self).__init__()
        
        # Convolutional Layers
        self.cnn_layer = nn.Sequential(
            # Conv1 Block
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),   # 16 x 224 x 224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                          # 16 x 112 x 112
            
            # Conv2 Block
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),                              # 32 x 112 x 112
            
            # Conv3 Block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # 64 x 112 x 112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                          # 64 x 56 x 56
            
            # Conv4 Block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 128 x 56 x 56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1)                               # 128 x 56 x 56
        )


         # Fully Connected Layers
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=128*56*56, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes)
        )
        
    def forward(self, x):
        x = self.cnn_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
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
    def __init__(self, num_classes=82, dropout_p=0.5, input_size=(1, 224, 224)):
        super(NetCNN_6, self).__init__()
        
        # Convolutional Layers
        self.cnn_layer = nn.Sequential(
            # Conv1 Block
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0),   # 16 x 222 x 222
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Conv2 Block
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),  # 32 x 110 x 110
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),                                         # 32 x 55 x 55
            
            # Conv3 Block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),  # 64 x 53 x 53
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Conv4 Block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0), # 128 x 26 x 26
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),                                         # 128 x 13 x 13
            
            # Conv5 Block
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),# 256 x 11 x 11
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)                                          # 256 x 5 x 5
        )
        
        # Fully Connected Layers
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=256 * 5 * 5, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
        
    def forward(self, x):
        x = self.cnn_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
        return x
    
    
# ResNet Basic Block 
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
# Customized Shallow Resnet
class ResNet(nn.Module):
  def __init__(self, num_classes=82):
    super(ResNet, self).__init__()
    self.in_channels = 16

    self.model = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      self._make_layer(16, 2, stride=1),
      self._make_layer(32, 2, stride=2),
      self._make_layer(64, 2, stride=2),
      self._make_layer(128, 2, stride=2),
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(start_dim=1),
      nn.Linear(128, len(classes))
  )
    self._initialize_weights()

  def _make_layer(self, out_channels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(BasicBlock(self.in_channels, out_channels, stride))
      self.in_channels = out_channels
    return nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)