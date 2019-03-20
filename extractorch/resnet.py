# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from torchvision import models

from .config import DATA_ROOT


class ResNet:
    def __init__(self, model_type, layer):
        self.model_type = model_type

        # Strip out '_places365'
        self.model_class = self.model_type.split('_')[0].lower()
        self.layer = layer
        self.num_classes = self.get_num_classes()
        self.pretrained = False if model_type.endswith('365') else True

        klass = getattr(models, self.model_class)
        print('Creating CNN instance {}'.format(self.model_type))
        self.model = klass(
            pretrained=self.pretrained, num_classes=self.num_classes)

        if self.num_classes == 365:
            self.load_places365_weights()

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

    def get_num_classes(self):
        if self.model_type.endswith('places365'):
            return 365
        return 1000

    def load_places365_weights(self):
        fname = DATA_ROOT / '{}.pth.tar'.format(self.model_type)
        weights = torch.load(
            fname, map_location=lambda st, loc: st)['state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in weights.items()}
        self.model.load_state_dict(state_dict)

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
            return (self.num_classes,)
        else:
            dummy = torch.zeros(1, 3, width, height, device='cpu')
            out = self.__extractor(dummy)
            return out.size()[1:]
