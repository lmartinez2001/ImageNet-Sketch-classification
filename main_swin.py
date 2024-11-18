from comet_ml import Experiment
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader


# from dataset import ImageNetDataset

from torch.optim.lr_scheduler import CosineAnnealingLR

# from model_factory import ModelFactory
from data import train_data_transforms, val_data_transforms
from model import Net

# Logging
import yaml

def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,

        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="number of warmup epochs",
    )
    parser.add_argument(
        "--epochs_before_decay",
        type=int,
        default=3,
        help="number of epoch before starting learning rate decay",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default='dinos',
        help="backbone to use as a feature extractor",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=10e-4,
        help="weight decay for SGD",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default='default_experiment',
        help="diplay name of the experiment in CometML",
    )
    parser.add_argument(
        '--finetune',
        action='store_true',
        help='set to finetune mode if enabled'
    )
    parser.add_argument(
        '--backbone_lr',
        type=float,
        help='learning rate for backbone finetuning'
    )
    parser.add_argument(
        '--cls_lr',
        type=float,
        help='classifier learning rate for finetuning'
    )
    parser.add_argument(
        '--model_ckpt_path',
        type=str,
        help='path of the checkpoint to finetune'
    )
    parser.add_argument(
        '--from_checkpoint',
        type=str,
        help='path of the checkpoint not to restart training form scratch in case of cracsh'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='gpu index to use'
    )
    args = parser.parse_args()
    return args

def warmup_lr(total_step, target_step, args):
    lr = min((total_step+1)/target_step, 1.0) * args.lr
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
    target_step = n_batches * args.warmup_epochs
    
    for batch_idx, (data, target) in enumerate(train_loader):
        total_step = batch_idx + (epoch-1) * n_batches
        if total_step < target_step and not args.finetune:
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr(total_step, target_step, args)

        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss(reduction="mean")
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


def validation(model: nn.Module, val_loader: torch.utils.data.DataLoader, use_cuda: bool, logger: Experiment) -> float:
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
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        
        criterion = nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        
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

    torch.cuda.set_device(args.device)
    # Logging
    logger = Experiment(
        api_key=private['comet']['key'],
        project_name=private['comet']['project'],
        workspace=private['comet']['workspace'],
    )
    logger.set_name(args.experiment_name)
    logger.log_parameters(vars(args))
    
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    model = Net(backbone=args.backbone, finetune=args.finetune)
    if args.finetune:
        print(f'Loaded {args.model_ckpt_path} state dict')
        model.load_state_dict(torch.load(args.model_ckpt_path, weights_only=True))
    
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")


    # Data initialization and loading
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/train_images", transform=train_data_transforms),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/val_images", transform=val_data_transforms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

   
    # Optimizer
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(model.backbone.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.finetune:
        print('Finetuning')
        optimizer = optim.AdamW([
            {'params': model.backbone.dinov2.parameters(), 'lr': args.backbone_lr},
            {'params': model.backbone.classifier.parameters(), 'lr': args.cls_lr}
        ])

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs-args.epochs_before_decay), eta_min=0.002)
    if args.finetune:
        scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs))

    start_epoch = 1
    if args.from_checkpoint:
        print(f'Restoring from state file {args.from_checkpoint}')
        checkpoint = torch.load(args.from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        
    # Loop over the epochs
    best_val_loss = 1e8
    for epoch in range(start_epoch, args.epochs + 1):
        
        if epoch > args.epochs_before_decay or args.finetune:
            print('Updating scheduler')
            scheduler.step()
        if not args.finetune:
            logger.log_metric('learning_rate', optimizer.param_groups[0]['lr'])
        
        train(model, optimizer, train_loader, use_cuda, epoch, args, logger)
        
        val_loss = validation(model, val_loader, use_cuda, logger)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_file = args.experiment + "/model_best_ft.pth" if args.finetune else args.experiment + "/model_best.pth"
            torch.save(model.state_dict(), best_model_file)
            # logger.log_model('best_model', best_model_file)
            
        # also save the model every epoch
        model_file = args.experiment + "/model_" + str(epoch) + "_ft.pth" if args.finetune else args.experiment + "/model_" + str(epoch) + ".pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': current_epoch,
        }, model_file)
        print(
            "Saved model to "
            + model_file
            + f". You can run `python evaluate.py --model_name {args.model_name} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )


if __name__ == "__main__":    
    main()
