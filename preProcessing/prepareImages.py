import csv
import glob
import time
from defisheye import Defisheye
import os
import random
from multiprocessing import Pool
from tqdm import tqdm

# Function to process a single image
def process_image(row):
    # Find the image file
    image_files = glob.glob(f"{images_dir_path}/{row['id']}.*")
    if not image_files:
        return  # Skip this row if no matching files were found

    image_file_path = image_files[0]

    # Set up defisheye with new parameters
    dtype = 'stereographic'  # type of fisheye lens: linear, equalarea, orthographic, stereographic
    format = 'fullframe'  # format of fisheye image: fullframe, circular
    fov = 160  # field of view of fisheye image in degrees (0 < fov <= 180)
    pfov = 100  # field of view of perspective image in degrees (0 < pfov < 180)

    obj = Defisheye(image_file_path, dtype=dtype, format=format, fov=fov, pfov=pfov)

    location = random.uniform(0, 1)
    output_dir_path = output_test_path if location<train_size+test_size else output_val_path
    if location<train_size: output_dir_path = output_train_path

    # Convert the image and save it to a new file
    output_image_file_path = f"{output_dir_path}/{row['country']}/{row['id']}.jpg"
    dir_name = os.path.dirname(output_image_file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    obj.convert(outfile=output_image_file_path)
    if not os.path.exists(f"{output_train_path}/{row['country']}"):
        os.makedirs(f"{output_train_path}/{row['country']}")
    if not os.path.exists(f"{output_test_path}/{row['country']}"):
        os.makedirs(f"{output_test_path}/{row['country']}")
    if not os.path.exists(f"{output_val_path}/{row['country']}"):
        os.makedirs(f"{output_val_path}/{row['country']}")


# Set the percentage of images to copy
percentage_to_copy = 1

# Set the paths to your files
csv_file_path = 'Data/archive/images.csv'
images_dir_path = 'Data/archive/images'
output_test_path = 'Data/labledCountries/test'
output_val_path = 'Data/labledCountries/val'
output_train_path = 'Data/labledCountries/train'

train_size = 0.7
val_size = 0.15
test_size = 0.15

# Read the CSV file
with open(csv_file_path, 'r') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Calculate the number of images to copy
num_images_to_copy = int(len(rows) * percentage_to_copy)

if __name__ == '__main__':
    # Create a pool of processes and distribute the task among them
    with Pool() as p:
        for _ in tqdm(p.imap_unordered(process_image, rows[:num_images_to_copy]), total=num_images_to_copy):
            pass
