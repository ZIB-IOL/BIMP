# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining - IOL Lab @ ZIB
# File:         config.py
# Description:  Datasets, Normalization and Transforms
# ===========================================================================

import torchvision
from torchvision import transforms

means = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'imagenet': (0.485, 0.456, 0.406),
}

stds = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'imagenet': (0.229, 0.224, 0.225),
}

datasetDict = {  # Links dataset names to actual torch datasets
    'mnist': getattr(torchvision.datasets, 'MNIST'),
    'cifar10': getattr(torchvision.datasets, 'CIFAR10'),
    'fashionMNIST': getattr(torchvision.datasets, 'FashionMNIST'),
    'SVHN': getattr(torchvision.datasets, 'SVHN'),  # This needs scipy
    'STL10': getattr(torchvision.datasets, 'STL10'),
    'cifar100': getattr(torchvision.datasets, 'CIFAR100'),
    'imagenet': getattr(torchvision.datasets, 'ImageNet'),
}

trainTransformDict = {  # Links dataset names to train dataset transformers
    'mnist': transforms.Compose([transforms.ToTensor()]),
    'cifar10': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar10'], std=stds['cifar10']), ]),
    'cifar100': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar100'], std=stds['cifar100']), ]),
    'imagenet': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['imagenet'], std=stds['imagenet']), ]),
}
testTransformDict = {  # Links dataset names to test dataset transformers
    'mnist': transforms.Compose([transforms.ToTensor()]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar10'], std=stds['cifar10']), ]),
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar100'], std=stds['cifar100']), ]),
    'imagenet': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['imagenet'], std=stds['imagenet']), ]),
}
