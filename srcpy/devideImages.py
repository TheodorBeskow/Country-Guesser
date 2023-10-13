import os
import json
from PIL import Image
import time 

with open('annotation_file_data3.json', 'r') as f:
    data = json.load(f)

src_dir = 'Data\\imagesVal'
dst_dir_true = 'Data\\LabeledImages\\val\\stopSigns'
dst_dir_false = 'Data\\LabeledImages\\val\\noStopSigns'

standard_size = (128, 128)
sinceTimeUpdate = time.time()

for i, filename in enumerate(os.listdir(src_dir)):
    if time.time()-sinceTimeUpdate > 3: 
        print(f"{i/len(os.listdir(src_dir))*100}% progress")
        sinceTimeUpdate = time.time()
    name = filename[:-4]
    if name + '.json' in data:
        img = Image.open(os.path.join(src_dir, filename))
        img_resized = img.resize(standard_size)
        if data[name + '.json']:
            img_resized.save(os.path.join(dst_dir_true, filename), 'JPEG', quality=95)
        else:
            img_resized.save(os.path.join(dst_dir_false, filename), 'JPEG', quality=95)
