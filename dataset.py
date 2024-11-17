import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes, pil_loader

import polars as pl
from data import val_data_transforms

class ImageNetDataset(Dataset):

    def __init__(self, root_dir, top_feature_file, transform):
        _, self.class_to_idx = find_classes(root_dir)
        df_features = pl.read_parquet(top_feature_file)

        self.root_dir = root_dir
        self.transform = transform
        
        # Topological features
        self.top_features = df_features['topology'].to_torch()
        self.top_features = torch.nan_to_num(self.top_features, nan=0.0)

        # class names = directories
        self.class_names = df_features['class_name'].to_numpy()

        # image file names
        self.im_names = df_features['image_name'].to_numpy()

    def __len__(self):
        return self.top_features.shape[0]

    def __getitem__(self, idx):

        class_name = self.class_names[idx]
        im_name = self.im_names[idx]
        im_path = os.path.join(self.root_dir, class_name, im_name)
        im = pil_loader(im_path)
        im = self.transform(im)
        
        top_feature = self.top_features[idx]
        
        label = self.class_to_idx[class_name]
        
        return im, top_feature, label




# TEST
if __name__ == '__main__':

    dataset = ImageNetDataset(root_dir='data/train_images', top_feature_file='topology/train_64_224px_final.parquet', transform=val_data_transforms)
    print(f'Dataset length {len(dataset)}')
    print(f'Shape of a sample {dataset[0][0].shape}')
    print(f'Associated label {dataset[0][2]}')
    print(f'Topological feature size {dataset[0][1].shape}')