import torch
import torch.nn as nn


class vgg16_bn_Mahdi(nn.Module):

    def __init__(self, numClasses):
        super(vgg16_bn_Mahdi, self).__init__()
        in_channels = 3

        self.features = nn.Sequential(

            nn.Conv2d(in_channels, 32, kernel_size=7, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32), #or 1d?
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False),
            #nn.Dropout2d(0.3),

            nn.Conv2d(32, 32, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32), #or 1d?
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False),

            nn.Conv2d(32, 128, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128), #or 1d?
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False),
            #nn.Dropout2d(0.4),

            nn.Conv2d(128, 256, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256), #or 1d?
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False),

            nn.Conv2d(256, 128, kernel_size=7, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128), #or 1d?
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False),
            #nn.Dropout2d(0.4),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256), #or 1d?
            #nn.MaxPool2d(kernel_size=4, stride=4, padding=1, dilation=1, ceil_mode=False),
            nn.MaxPool2d(kernel_size=7, stride=4, padding=1, dilation=1, ceil_mode=False),
            #nn.Dropout(0.5),

        )

        self.classifierG = nn.Sequential(

            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, numClasses),

            #nn.Sigmoid()

        )

        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                #nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        for layer in self.classifierG:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                #nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    # Defining the forward pass
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifierG(x)

        return x


