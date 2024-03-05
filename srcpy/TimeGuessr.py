from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import json
import os
import random

options = Options()
options.headless = True 
driver = webdriver.Chrome(options=options)
driver.get('https://timeguessr.com/roundone')  

while True:


    time.sleep(random.uniform(1, 2))  

    play_array_json = driver.execute_script("return sessionStorage.getItem('playArray');")

    play_array = json.loads(play_array_json)
    file_path = 'Data/TimeGuessr/data.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []

    counter = 0
    with open(file_path, 'w') as f:
        for item in play_array:
            if isinstance(item, dict) and 'URL' in item:
                if item in existing_data:
                    counter += 1
                else:
                    existing_data.append(item)
        json.dump(existing_data, f)

    print("Duplicates:", counter)
    
    # Refresh the page
    driver.refresh()

# Quit the driver after the loop
driver.quit()
