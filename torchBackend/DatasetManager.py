from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
from PIL import Image
import os
import torch

def dataLoaderGenerator(dataset, train_ids, val_ids,batch_size):
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

    return train_loader, validation_loader


class MicroplastDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = f"{self.annotations.iloc[index, 0]}_P.bmp"
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform is not None:
            img = self.transform(img)

        return img, y_label