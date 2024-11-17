"""Python file to instantite the model and the transform that goes with it."""

from data import train_data_transforms, val_data_transforms
from model import DinoNet, SwimNEt


class ModelFactory:
    def __init__(self, model_name: str, n_layers, backbone, topological_resolution):
        self.model_name = model_name
        self.n_layers = n_layers
        self.backbone = backbone
        self.topological_resolution = topological_resolution
        self.model = self.init_model()
        self.train_transform = self.init_train_transform()
        self.val_transform = self.init_val_transform()

    def init_model(self):
        if self.model_name == "dino":
            return DinoNet(n_layers=self.n_layers, backbone=self.backbone, topological_resolution=self.topological_resolution)
        elif self.model_name == 'swim':
            return SwimNet()
        else:
            raise NotImplementedError("Model not implemented")

    def init_train_transform(self):
        return train_data_transforms
         
    def init_val_transform(self):
        return val_data_transforms


    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.train_transform, self.val_transform
