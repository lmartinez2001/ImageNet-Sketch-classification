import argparse
import os

import PIL.Image as Image
import torch
from tqdm import tqdm

import torchvision.transforms as T
from model_factory import ModelFactory
import polars as pl

val_data_transforms = T.Compose([
        T.Resize((140,140)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 evaluation script")
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        metavar="D",
        help="folder where data is located. test_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=1,
        metavar="NL",
        help="Number of layers of the MLP classifier",
    )
    parser.add_argument(
        "--model",
        type=str,
        metavar="M",
        help="the model file to be evaluated. Usually it is of the form model_X.pth",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="dino",
        metavar="MOD",
        help="Name of the model for model and transform instantiation.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="experiment/kaggle.csv",
        metavar="D",
        help="name of the output csv file",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="dino",
        metavar="BB",
        help="backbone model",
    )
    parser.add_argument(
        "--topological_resolution",
        type=int,
        default=64,
        metavar="TR",
        help="Resolution of the precomputed topological features",
    )
    args = parser.parse_args()
    return args


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def main() -> None:
    """Main Function."""
    # options
    args = opts()
    test_dir = args.data + "/test_images/mistery_category"
    features_df = pl.read_parquet('topology/test_res64_224px_final.parquet')

    imgs = features_df['image_name'].to_numpy()
    features = features_df['topology'].to_torch()
    features = torch.nan_to_num(features)
    features_dict = dict(zip(imgs, features))
    # cuda
    use_cuda = torch.cuda.is_available()

    # load model and transform
    state_dict = torch.load(args.model)
    model, _, val_data_transforms = ModelFactory(
        args.model_name, 
        n_layers=args.n_layers, 
        backbone=args.backbone,
        topological_resolution=args.topological_resolution).get_all()
    model.load_state_dict(state_dict)
    model.eval()
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    output_file = open(args.outfile, "w")
    output_file.write("Id,Category\n")
    for f in tqdm(os.listdir(test_dir)):
        if "jpeg" in f:
            data = val_data_transforms(pil_loader(test_dir + "/" + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            top_features = features_dict[f].unsqueeze(0)
            if use_cuda:
                data = data.cuda()
                top_features = top_features.cuda()
            output = model(data, top_features)
            pred = output.data.max(1, keepdim=True)[1]
            output_file.write("%s,%d\n" % (f[:-5], pred))

    output_file.close()

    print(
        "Succesfully wrote "
        + args.outfile
        + ", you can upload this file to the kaggle competition website"
    )


if __name__ == "__main__":
    main()
