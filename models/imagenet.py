# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining
# File:         imagenet.py
# Description:  ImageNet Models
# ===========================================================================

import torchvision


def ResNet50():
    return torchvision.models.resnet50(pretrained=False)
