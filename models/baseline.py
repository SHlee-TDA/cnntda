import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineImgConv2d(nn.Module):
    def __init__(self, input_shape=(3, 51, 51)):
        super(BaselineImgConv2d, self).__init__()
        in_channels, h, w = input_shape
        self.block1 = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding='same'),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=2)
                    )
        
        self.block2 = nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                    )
        
        self.block3 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
        )

        self.block4 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same'),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        # Process image input
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        return x


class BaselineTopoConv1d(nn.Module):
    def __init__(self, input_shape=(1, 100)):
        super(BaselineTopoConv1d, self).__init__()
        in_channels, l = input_shape
        
        def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )

        def skip_connection(in_channels, out_channels):
            return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

        self.skip1 = skip_connection(in_channels, 32)
        self.conv1 = conv_bn_relu(in_channels, 32)
        
        self.skip2 = skip_connection(32, 64)
        self.conv2 = conv_bn_relu(32, 64)
        
        self.conv3 = conv_bn_relu(64, 128)
        self.skip3 = skip_connection(64, 128)

        self.conv4 = conv_bn_relu(128, 256)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # First block
        residual = self.skip1(x)
        x = self.conv1(x)
        x = torch.add(residual, x)  # Add skip-connection
        x = self.pool(x)

        # Second block
        residual = self.skip2(x)
        x = self.conv2(x)
        x = torch.add(residual, x)  # Add skip-connection
        x = self.pool(x)

        # Third block
        residual = self.skip3(x)
        x = self.conv3(x)
        x = torch.add(residual, x)  # Add skip-connection
        x = self.pool(x)

        # Fourth block
        x = self.conv4(x)
        x = self.pool(x)

        return x.view(x.size(0), -1)
        

class BaselineMlpClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BaselineMlpClassifier, self).__init__()

        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc_drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = F.relu(x)
        x = self.fc_drop1(x)
        
        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        #x = self.softmax(x, dim=1)
        
        return x
    

class BaselineCNN(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(BaselineCNN, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        feature_map = self.feature_extractor(x)
        output = self.classifier(feature_map)
        return output

    @property
    def module(self):
        """
        If the model is being used with DataParallel, 
        this property will return the actual model.
        """
        return self._modules.get('module', self)
