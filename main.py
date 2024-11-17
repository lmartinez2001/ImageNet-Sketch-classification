from comet_ml import Experiment
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader

from dataset import ImageNetDataset

from torch.optim.lr_scheduler import CosineAnnealingLR

from model_factory import ModelFactory

# Logging
import yaml

def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        metavar="NW",
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        metavar="WS",
        help="number of warmup steps",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=1,
        metavar="NL",
        help="number hidden layers for the classifier",
    )
    parser.add_argument(
        "--epochs_before_decay",
        type=int,
        default=15,
        metavar="EBD",
        help="number of epoch before starting learning rate decay",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default='dino',
        metavar="BB",
        help="backbone to use as a feature extractor",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=10e-4,
        metavar="WD",
        help="weight decay for SGD",
    )
    parser.add_argument(
        "--topological_resolution",
        type=int,
        default=64,
        metavar="TR",
        help="size of the topological features vector",
    )
    parser.add_argument(
        "--train_top_features_file",
        type=str,
        default='topology/train_64_224px_final.parquet',
        metavar="TTF",
        help="dataframe containing the topological features for each image",
    )
    parser.add_argument(
        "--val_top_features_file",
        type=str,
        default='topology/val_64_224px_final.parquet',
        metavar="VTF",
        help="dataframe containing the topological features for each image",
    )
    args = parser.parse_args()
    return args

def warmup_lr(total_step, args):
    lr = min((total_step+1)/args.warmup_steps, 1.0) * args.lr
    return lr


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    args: argparse.ArgumentParser,
    logger: Experiment
) -> None:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
    """
    model.train()
    correct = 0
    epoch_loss = 0
    n_batches = len(train_loader)
    
    for batch_idx, (data, top_features, target) in enumerate(train_loader):
        total_step = batch_idx + (epoch-1) * n_batches
        if total_step < args.warmup_steps:
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr(total_step, args)

        if use_cuda:
            data, top_features, target = data.cuda(), top_features.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data, top_features)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        epoch_loss += loss.data.item()
        
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
    accuracy = 100.0 * correct / len(train_loader.dataset)
    
    logger.log_metric('train_loss', epoch_loss / len(train_loader.dataset))
    logger.log_metric('train_accuracy', accuracy)  
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            accuracy,
        )
    )


def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    logger: Experiment
) -> float:
    """Default Validation Loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        float: Validation loss
    """
    model.eval()
    validation_loss = 0
    correct = 0
    for data, top_features, target in val_loader:
        if use_cuda:
            data, top_features, target = data.cuda(), top_features.cuda(), target.cuda()
        output = model(data, top_features)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    accuracy = 100.0 * correct / len(val_loader.dataset)

    logger.log_metric('validation_loss', validation_loss)
    logger.log_metric('validation_accuracy', accuracy)
    
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            accuracy,
        )
    )
    return validation_loss


def main():
    """Default Main Function."""
     # Private info
    with open('private.yml', 'r') as f:
        private = yaml.load(f, Loader=yaml.SafeLoader)
    
    # options
    args = opts()

    # Logging
    logger = Experiment(
        api_key=private['comet']['key'],
        project_name=private['comet']['project'],
        workspace=private['comet']['workspace'],
    )

    logger.log_parameters(vars(args))
    
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    # use_cuda = False

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # load model and transform
    model, train_data_transforms, val_data_transforms = ModelFactory(
        model_name=args.model_name, 
        n_layers=args.n_layers, 
        backbone=args.backbone, 
        topological_resolution=args.topological_resolution
    ).get_all()
    
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    train_set = ImageNetDataset(
        root_dir='data/train_images',
        # top_feature_file='topology/train_64_224px_final.parquet',
        top_feature_file=args.train_top_features_file,
        transform=train_data_transforms
    )

    val_set = ImageNetDataset(
        root_dir='data/val_images',
        # top_feature_file='topology/val_64_224px_final.parquet',
        top_feature_file=args.val_top_features_file,
        transform=val_data_transforms
    )

    # Data initialization and loading
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(args.data + "/train_images", transform=train_data_transforms),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(args.data + "/val_images", transform=val_data_transforms),
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    # )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs-args.epochs_before_decay))

    # Loop over the epochs
    best_val_loss = 1e8
    for epoch in range(1, args.epochs + 1):
        if epoch > args.epochs_before_decay:
            scheduler.step()

        logger.log_metric('learning_rate', optimizer.param_groups[0]['lr'])
        # training loop
        train(model, optimizer, train_loader, use_cuda, epoch, args, logger)
        
        # validation loop
        val_loss = validation(model, val_loader, use_cuda, logger)
        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            best_model_file = args.experiment + "/model_best.pth"
            torch.save(model.state_dict(), best_model_file)
            # logger.log_model('best_model', best_model_file)
            
        # also save the model every epoch
        model_file = args.experiment + "/model_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), model_file)
        print(
            "Saved model to "
            + model_file
            + f". You can run `python evaluate.py --model_name {args.model_name} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )


if __name__ == "__main__":    
    main()
