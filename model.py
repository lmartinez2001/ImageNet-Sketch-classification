import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import ImageNetDataset
from data import val_data_transforms
from torch.utils.data import DataLoader 
from transformers import AutoModelForImageClassification
       

class Net(nn.Module):
    def __init__(self, n_classes: int = 500, layers: list = [1152, 1152], backbone: str = 'dinos', finetune: bool = False):
        """
        backbone: str
            Choice of feature extractor
            Valid options are: dinos, dinob, 
        """
        assert len(layers) > 0
        super().__init__()
        self.layers = layers
        
        # Choosing the dino backbone among all available model sizes (except the largest one which is to big)
        backbone_options = {
            'dinos': 'facebook/dinov2-small',
            'dinob': 'facebook/dinov2-base',
            'dinol': 'facebook/dinov2-large',
            'convnext': 'facebook/convnextv2-base-1k-224',
            'swin': 'microsoft/swinv2-small-patch4-window16-256'
        }
        if backbone not in backbone_options:
            raise ValueError(f"Invalid backbone option: {backbone}. Choose 'dinos' or 'dinob'")

        model_name = backbone_options[backbone]
        self.backbone = AutoModelForImageClassification.from_pretrained(model_name)
        print(f'Using {model_name} backbone')

        # Number of output feartures used as inputs for the classification head
        if backbone == 'dinos' or backbone == 'dinob' or backbone == 'dinol':  
            n_backbone_features: int = 2 * self.backbone.dinov2.layernorm.normalized_shape[0]
            for param in self.backbone.dinov2.parameters():
                param.requires_grad = finetune
        elif backbone == 'convnext':
            n_backbone_features: int = self.backbone.convnextv2.layernorm.normalized_shape[0]
            for param in self.backbone.convnextv2.parameters():
                param.requires_grad = finetune
        elif backbone == 'swin':
            n_backbone_features: int = self.backbone.swinv2.layernorm.normalized_shape[0]
            for param in self.backbone.swinv2.parameters():
                param.requires_grad = finetune
        
        # Freeze backbone
        

        self.backbone.classifier = self.make_classifier(n_backbone_features, n_classes, layers)

    
    def forward(self, x):
        return self.backbone(x).logits

    def make_classifier(self, in_features, n_classes: int, layers: list):
        classifier = [self._make_linear_block(in_features, layers[0])] 
        if len(layers) > 1:
            classifier += [self._make_linear_block(layers[i], layers[i+1]) for i in range(len(layers)-1)] 
        classifier += [nn.Linear(layers[-1], n_classes)]
        return nn.Sequential(*classifier)

    def _make_linear_block(self, in_features, out_features, dropout=0.5):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        

# FOR TESTING
if __name__ == '__main__':
    device = 'cuda:0'
    dataset = ImageNetDataset(
        root_dir='data/train_images',
        top_feature_file='topology/train_64_224px_final.parquet',
        transform=val_data_transforms
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True
    )
    
    im, top_features, label = next(iter(dataloader))
    im = im.to(device)
    top_features = top_features.to(device)
    
    net = Net(
        topological_resolution=64,
        n_layers=2, 
        n_classes=500
    ).to(device)
    
    out = net(im, top_features).detach()
    print(out.shape)
