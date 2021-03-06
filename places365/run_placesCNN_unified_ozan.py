#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017
# last modified date: Dec. 27, 2017, migrating everything to python36 and latest pytorch and torchvision

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable as V
from torch.nn import functional as F
import torchvision.models as models
from torchvision import transforms as trn

from PIL import Image
import cv2


MODEL_PATH = Path('/disk2/tmp/caglayan/data/places-cnn')


def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute


def returnCAM(feature_conv, weight_softmax, class_idx):
    def normalize(x, uint=False):
        xx = (x - x.min()) / (x.max() - x.min())
        if uint:
            return np.uint8(255 * xx)
        else:
            return xx

    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        # reshape back to h*w
        cam = normalize(cam.reshape(h, w))
        cam = cv2.resize(cam, size_upsample)
        output_cam.append(normalize(cam, True))
    return output_cam


def hook_feature(module, input, output):
    module.blob = np.squeeze(output.data.cpu().numpy())


def load_model(useGPU=0):
    # this model has a last conv feature map as 14x14

    model_file = MODEL_PATH / 'whole_wideresnet18_places365_python36.pth.tar'
    if not model_file.exists():
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)

    if useGPU == 1:
        model = torch.load(model_file)
    else:
        model = torch.load(
            model_file,
            map_location=lambda storage, loc: storage) # allow cpu

    model.eval()

    # hook the feature extractor
    # this is the last conv layer of the resnet
    features_names = ['layer4', 'avgpool']

    for name in features_names:
        submodule = model._modules.get(name)
        submodule.blob = None
        submodule.register_forward_hook(hook_feature)
    return model


if __name__ == '__main__':
    try:
        test_img = Path(sys.argv[1])
    except IndexError as ie:
        print('Usage: {} <test_img_file>'.format(sys.argv[0]))
        sys.exit(1)

    # load the labels
    classes, labels_IO, labels_attribute, W_attribute = load_labels()

    # load the transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the model
    model = load_model()

    # load the test image
    img = Image.open(test_img)
    input_img = V(tf(img).unsqueeze(0), volatile=True)

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output the prediction of scene category
    print('--SCENE CATEGORIES:')
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # output the scene attributes
    responses_attribute = W_attribute.dot(model.avgpool.blob)
    idx_a = np.argsort(responses_attribute)
    print('--SCENE ATTRIBUTES:')
    print(', '.join([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]))

    # generate class activation mapping
    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    #weight_softmax[weight_softmax < 0] = 0
    print(model.layer4.blob.shape)
    CAMs = returnCAM(model.layer4.blob, weight_softmax, [idx[0]])

    # render the CAM and output
    img = cv2.imread(str(test_img))
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite("CAM_" + test_img.name, result)
