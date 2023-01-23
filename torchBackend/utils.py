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

    if desired_model == 'models.alexnet()':
        model = models.alexnet()
        model.classifier[6] = nn.Linear(4096, num_classes)

    if desired_model == 'models.resnet18()':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model
