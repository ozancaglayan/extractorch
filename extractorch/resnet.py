# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from torchvision import models


class ResNet:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer

        klass = getattr(models, self.model.lower())
        print('Creating CNN instance {}'.format(self.model))
        self.model = klass(pretrained=True)

        # Replace avgpool with adaptive one to support variable input sizes
        del self.model.avgpool
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.layer != 'prob':
            # Remove final classifier layer
            del self.model.fc

        # Set extractor
        self.__extractor = getattr(self, self.layer)

        # Turn on eval mode
        self.model.train(False)

    def set_device(self, device):
        self.model.to(device)

    def base_forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        return x

    def res4f_relu(self, x):
        return self.model.layer3(self.base_forward(x))

    def res5c_relu(self, x):
        return self.model.layer4(self.res4f_relu(x))

    def avgpool(self, x):
        x = self.model.avgpool(self.res5c_relu(x))
        return x.view(x.size(0), -1)

    def prob(self, x):
        return F.softmax(self.model(x), dim=1)

    def __call__(self, x):
        return self.__extractor(x)

    def output_shape(self, width, height):
        if self.layer == 'avgpool':
            return (2048,)
        elif self.layer == 'prob':
            return (1000,)
        else:
            dummy = torch.zeros(1, 3, width, height, device='cpu')
            out = self.__extractor(dummy)
            return out.size()[1:]
