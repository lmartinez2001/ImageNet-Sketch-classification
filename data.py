import torchvision.transforms as T

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from ImageNet

# train_data_transforms = T.Compose([
#         T.RandomPerspective(distortion_scale=0.3),
#         # T.RandomResizedCrop(140,scale=(0.5,1.0)),
#         T.Resize((224,224)),
#         T.RandomHorizontalFlip(p=0.5),
#         T.RandomVerticalFlip(p=0.5),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# val_data_transforms = T.Compose([
#         T.Resize((224,224)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


train_data_transforms = T.Compose([
        T.RandomPerspective(distortion_scale=0.3),
        T.RandomResizedCrop(224,scale=(0.5,1.0)),
        # T.Resize((224,224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


val_data_transforms = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])   


