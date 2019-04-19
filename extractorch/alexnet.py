# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models import alexnet


class AlexNet:
    def __init__(self, **kwargs):
        print('Creating CNN instance "alexnet"')
        self.model = alexnet(True)

        # Turn on eval mode
        self.model.train(False)

    def set_device(self, device):
        self.model.to(device)

    def __call__(self, x):
        return self.model.features(x)

    def output_shape(self, width, height):
        dummy = torch.zeros(1, 3, width, height, device='cpu')
        out = self(dummy)
        return out.size()[1:]
