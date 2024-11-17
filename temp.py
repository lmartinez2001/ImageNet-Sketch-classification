import os

min_number = 100000
max_number = 0

for f in os.listdir('data/train_images'):
    num = len(os.listdir(os.path.join('data', 'train_images', f)))
    if num > max_number:
        max_number = num
    if num < min_number:
        min_number = num

print(min_number, max_number)