import logging

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn


def getTransformer(transform_resize, transform_crop, transform_normalize_mean, transform_normalize_var):

    transform = transforms.Compose(
            [
                transforms.Resize(transform_resize),
                transforms.RandomCrop(transform_crop),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(transform_normalize_mean, transform_normalize_var),
            ]
        )

    return transform


def generateModel(desired_model, num_classes):

    if desired_model == 'alexNet':
        model = models.alexnet()
        model.classifier[6] = nn.Linear(4096, num_classes)

    if desired_model == 'resNet18':
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    if desired_model == 'vgg11':
        model = models.vgg11()

    if desired_model == 'mobilenet_v2':
        model = models.mobilenet_v2()

    if "model" in locals():
        return model
    else:
        logging.error(f'the name of the network {desired_model} is not in the available models list')
