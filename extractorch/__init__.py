#
from .alexnet import AlexNet
from .resnet import ResNet
from .vgg import VGG
from .densenet import DenseNet

def get_cnn(model_type):
    return {
        'alexnet': AlexNet,
        'vgg': VGG,
        'resnet': ResNet,
        'densenet': DenseNet,
    }[model_type]
