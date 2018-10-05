extractorch
---

`extractorch` is a feature extraction utility for Pytorch 0.4 and torchvision.

## Features

### Generic Features
 - Supports batched CPU/GPU extraction
 - Directly processes a folder of raw images with an `index.txt` file within
   that determines the list and the order of images to process.
 - Support for cropping with optional zoom (`--central-fraction`)
 - Optional L2 normalization for all features except classification scores
 - Output file is a `float16` `.npy` tensor.

### Supported CNNs

#### ResNet 18/34/50/101/152
  - res4f_relu, res5c_relu, avgpool and classification features.

#### Places365
  - Work in progress
