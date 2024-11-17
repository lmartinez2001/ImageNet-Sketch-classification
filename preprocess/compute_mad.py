import polars as pl
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def mean_absolute_deviation(image_array):
    return np.mean(np.abs(image_array - np.mean(image_array)))



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='../data/val_images')
parser.add_argument('--out', type=str, default='mad_val.parquet')
if __name__ == '__main__':
    args = parser.parse_args()
    data = []
    
    for class_name in tqdm(os.listdir(args.root)):
        class_dir = os.path.join(args.root, class_name)
        
        # Check if it's a directory
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                
                # Open and process each image as grayscale
                if 'jpeg' in image_path:
                    with Image.open(image_path) as img:
                        img_gray = img.convert('L')  # Convert to grayscale
                        image_array = np.array(img_gray)
                        mad = mean_absolute_deviation(image_array)
                    
                    # Append to data list
                    data.append((image_name, class_name, mad))
    
    # Create the Polars DataFrame
    df = pl.DataFrame(data, schema=['image_name', 'class_name', 'mean_absolute_deviation'])
    df.write_parquet(args.out)