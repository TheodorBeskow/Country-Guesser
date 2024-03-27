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
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = './ffmpeg'

def map_to_sphere(x, y, z, yaw_radian, pitch_radian):


    theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi = np.arctan2(y, x)

    # Apply rotation transformations here
    theta_prime = np.arccos(np.sin(theta) * np.sin(phi) * np.sin(pitch_radian) +
                            np.cos(theta) * np.cos(pitch_radian))

    phi_prime = np.arctan2(np.sin(theta) * np.sin(phi) * np.cos(pitch_radian) -
                           np.cos(theta) * np.sin(pitch_radian),
                           np.sin(theta) * np.cos(phi))
    phi_prime += yaw_radian
    phi_prime = phi_prime % (2 * np.pi)

    return theta_prime.flatten(), phi_prime.flatten()


def interpolate_color(coords, img, method='bilinear'):
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    return np.stack((red, green, blue), axis=-1)


def panorama_to_plane(panorama_path, FOV, output_size, yaw, pitch):
    panorama = Image.open(panorama_path).convert('RGB')
    width, height = panorama.size
    panorama = panorama.crop((0, 0, int(width*(height/256)), height))  

    pano_width, pano_height = panorama.size
    pano_array = np.array(panorama)
    yaw_radian = np.radians(yaw)
    pitch_radian = np.radians(pitch)

    W, H = output_size
    f = (0.5 * W) / np.tan(np.radians(FOV) / 2)

    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    x = u - W / 2
    y = H / 2 - v
    z = f

    theta, phi = map_to_sphere(x, y, z, yaw_radian, pitch_radian)

    U = phi * pano_width / (2 * np.pi)
    V = theta * pano_height / np.pi

    U, V = U.flatten(), V.flatten()
    coords = np.vstack((V, U))

    colors = interpolate_color(coords, pano_array)
    output_image = Image.fromarray(colors.reshape((H, W, 3)).astype('uint8'), 'RGB')

    return output_image

def ascii_country_name(country_name):
    # Transliterate to closest ASCII representation
    country_name = unidecode(country_name)
    return country_name

country_amount = defaultdict(int)
PER_COUNTRY = 5000

# using = ["United States of America", "France", "India", "Russian Federation", "Canada"]
using = ["United States of America", "France", "India", "Russian Federation", "Canada", "Brazil", "Australia", "Italy", "Japan", "Spain", "Thailand", "U.K. of Great Britain and Northern Ireland", "Poland", "Sweden", "Mexico"]

def process_image(row):
    if row['country'] not in using: return
    # if row['country'] != "Portugal" and row['country'] != "Denmark": return
    # if row['country'] != "Denmark" and row['country'] != "Sweden" and row['country'] != "Norway" and row['country'] != "Poland" and row['country'] != "Finland" and row['country'] != "Mexico": return


    amounts = random.uniform(math.floor(PER_COUNTRY/country_amount[row['country']]), math.floor(PER_COUNTRY/country_amount[row['country']]+1))
    realAmount = math.floor(PER_COUNTRY/country_amount[row['country']]) if amounts>PER_COUNTRY/country_amount[row['country']] else math.floor(PER_COUNTRY/country_amount[row['country']]+1)
    if realAmount==0: return

    image_files = glob.glob(f"{images_dir_path}/{row['id']}.*")
    if not image_files: return

    image_file_path = image_files[0]

    location = random.uniform(0, 1)
    output_dir_path = output_test_path if location<train_size+test_size else output_val_path
    if location<train_size: output_dir_path = output_train_path

    # print(row['country'],  PER_COUNTRY/country_amount[row['country']])
    for am in range(realAmount):
        if am >= 5: break
        # country = ascii_country_name(row['country'])
        country = row['country']
        output_image_file_path = f"{output_dir_path}/{country}/{row['id']}Num{am}.jpg"

        dir_name = os.path.dirname(output_image_file_path)
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
            except:
                pass
        output_image = panorama_to_plane(image_file_path, 110+random.randint(0, 20), (128, 128), 180+70+random.randint(0, 40), 80+random.randint(0, 30))
        output_image.save(output_image_file_path)

    # Ifall det är väldigt få!
    # if not os.path.exists(f"{output_train_path}/{row['country']}"):
    #     os.makedirs(f"{output_train_path}/{row['country']}")
    # if not os.path.exists(f"{output_test_path}/{row['country']}"):
    #     os.makedirs(f"{output_test_path}/{row['country']}")
    # if not os.path.exists(f"{output_val_path}/{row['country']}"):
    #     os.makedirs(f"{output_val_path}/{row['country']}")


# Set the percentage of images to copy
percentage_to_copy = 1

# Set the paths to your files
csv_file_path = 'Data/archive/images.csv'
images_dir_path = 'Data/archive/images'
output_test_path = 'Data/labledCountries128/test'
output_val_path = 'Data/labledCountries128/val'
output_train_path = 'Data/labledCountries128/train'

train_size = 0.8
val_size = 0.10
test_size = 0.10

# Read the CSV file
with open(csv_file_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

for index in range(len(rows)):
    country_amount[rows[index]['country']]+=1

# Calculate the number of images to copy
num_images_to_copy = int(len(rows) * percentage_to_copy)

if __name__ == '__main__':
    # Create a pool of processes and distribute the task among them
    with Pool() as p:
        for _ in tqdm(p.imap_unordered(process_image, rows[:num_images_to_copy]), total=num_images_to_copy):
            pass
