import csv
import glob
import time
from defisheye import Defisheye
import os
import random
from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict
import math
from unidecode import unidecode

def ascii_state_name(state_name):
    # Transliterate to closest ASCII representation
    state_name = unidecode(state_name)
    return state_name

country_amount = defaultdict(int)
PER_COUNTRY = 1000

def process_image(row):
    
    # if row['state'] != "United States of America": return
    # if row['state'] != "Denmark" and row['state'] != "Sweden" and row['state'] != "Norway" and row['state'] != "Poland" and row['state'] != "Finland" and row['state'] != "Mexico": return

    image_files = glob.glob(f"{images_dir_path}/{row['id']}.*")
    if not image_files: return

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

    amounts = random.uniform(math.floor(PER_COUNTRY/country_amount[row['state']]), math.floor(PER_COUNTRY/country_amount[row['state']]+1))
    # print(row['state'],  PER_COUNTRY/country_amount[row['state']])
    for am in range(math.floor(PER_COUNTRY/country_amount[row['state']]) if amounts>PER_COUNTRY/country_amount[row['state']] else math.floor(PER_COUNTRY/country_amount[row['state']]+1)):
        if am >= 10: break
        state = ascii_state_name(row['state'])
        output_image_file_path = f"{output_dir_path}/{state}/{row['id']}Num{am}.jpg"

        dir_name = os.path.dirname(output_image_file_path)
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
            except:
                pass
        obj.convert(outfile=output_image_file_path)

    # Ifall det är väldigt få!
    # if not os.path.exists(f"{output_train_path}/{row['state']}"):
    #     os.makedirs(f"{output_train_path}/{row['state']}")
    # if not os.path.exists(f"{output_test_path}/{row['state']}"):
    #     os.makedirs(f"{output_test_path}/{row['state']}")
    # if not os.path.exists(f"{output_val_path}/{row['state']}"):
    #     os.makedirs(f"{output_val_path}/{row['state']}")


# Set the percentage of images to copy
percentage_to_copy = 1

# Set the paths to your files
csv_file_path = 'Data/archive/brazil.csv'
images_dir_path = 'Data/archive/images'
output_test_path = 'Data/labledStates/test'
output_val_path = 'Data/labledStates/val'
output_train_path = 'Data/labledStates/train'

train_size = 0.8
val_size = 0.10
test_size = 0.10

# Read the CSV file
with open(csv_file_path, 'r') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

for index in range(len(rows)):
    country_amount[rows[index]['state']]+=1

# Calculate the number of images to copy
num_images_to_copy = int(len(rows) * percentage_to_copy)

if __name__ == '__main__':
    # Create a pool of processes and distribute the task among them
    with Pool() as p:
        for _ in tqdm(p.imap_unordered(process_image, rows[:num_images_to_copy]), total=num_images_to_copy):
            pass
