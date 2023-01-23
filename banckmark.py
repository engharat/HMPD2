import os
import torch
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
import json
from Utility.yamlManager import yamlManager
from torchBackend.utils import *
from torchBackend.DatasetManager import MicroplastDataset
from torchBackend.DatasetManager import dataLoaderGenerator
from torchBackend.trainTest import trainTest


if __name__ == "__main__":

    config_file = "./config/costuomConf.yaml"
    banckmark_name = 'test1'

    if not os.path.exists(f"./tests/{banckmark_name}"):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(f"./tests/{banckmark_name}")

    yM = yamlManager(config_file, f"./tests/{banckmark_name}/banchmarkconfiguration.yaml")
    configuration = yM.read_conf()

    num_epochs = configuration['num_epochs']
    learning_rate = configuration['learning_rate']
    train_CNN = configuration['train_CNN']
    batch_size = configuration['batch_size']
    shuffle = configuration['shuffle']
    pin_memory = configuration['pin_memory']
    num_workers = configuration['num_workers']
    transform_resize = configuration['transform_resize']
    transform_crop = configuration['transform_crop']
    transform_normalize_mean = configuration['transform_normalize_mean']
    transform_normalize_var = configuration['transform_normalize_var']
    listofNetwork: configuration['listofNetwork']

    transform = getTransformer(transform_resize, transform_crop, transform_normalize_mean, transform_normalize_var)
    device = 'cpu'
    kfold = KFold(n_splits=5, shuffle=True)
    dataset = MicroplastDataset("/Users/beppe2hd/Data/Microplastiche/images", "balanced.csv", transform=transform)

    for k, m in listofNetwork:

        model = generateModel(m)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        reportData = []
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f'FOLD {fold}')

            train_loader, validation_loader = dataLoaderGenerator(train_ids, val_ids)
            traintestfold = trainTest(model, device, criterion, optimizer)
            traintestfold.train(train_loader, validation_loader, num_epochs)
            val_acc, conf, predictions, yGT, probs = traintestfold.check_full_accuracy(validation_loader)

            probs_1 = [tensor[1].item() for tensor in probs]
            yGT = [tensor.item() for tensor in yGT]


            current_data = {'fold': fold, 'gt': yGT, 'probsClass1': probs_1}

            reportData.append(current_data)

        with open(f"./tests/{banckmark_name}/{k}.json", "w") as final:
            json.dump(reportData, final)


