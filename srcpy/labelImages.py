import os
import json
import time
import threading

directory = r'Data\mtsd_v2_fully_annotated\annotations'
file_names = os.listdir(directory)
file_names = [f for f in file_names if os.path.isfile(os.path.join(directory, f))]

file_data = {}
total_files = len(file_names)
processed_files = 0

def print_progress():
    while processed_files < total_files:
        print(f'Processed: {processed_files/total_files*100:.2f}%')
        time.sleep(3)

progress_thread = threading.Thread(target=print_progress)
progress_thread.start()

for file_name in file_names:
    with open(os.path.join(directory, file_name), 'r') as f:
        content = f.read()
        file_data[file_name] = 'regulatory--stop--g1' in content
    processed_files += 1

with open('annotation_file_data3.json', 'w') as f:
    json.dump(file_data, f)
