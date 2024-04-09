import tkinter as tk
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import sys

# Load the model
model = load_model(r'.\Data\trainedModel.h5')

# Directory containing the images
data_dir = r'.\Data\labledCountries128\test'

# List of class names
class_names = os.listdir(data_dir)

# List of image paths
image_paths = [os.path.join(data_dir, class_name, image_name)
               for class_name in class_names
               for image_name in os.listdir(os.path.join(data_dir, class_name))]

# Shuffle the list of image paths
random.shuffle(image_paths)

# Index of the current image
image_index = 0

# Read coordinate data from CSV
coordinate_data = {}
with open(r'Data\archive\images.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_id = row['id']
        coordinate_data[image_id] = (float(row['lat']), float(row['lng']))

auto_playing = False

def show_image():
    global image_index

    # Load the image
    image_path = image_paths[image_index]
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image).reshape((1, 128, 128, 3)) / 255.0

    # Make a prediction
    prediction = model.predict(image, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Get the actual class
    actual_class = os.path.basename(os.path.dirname(image_path))

    # Update the image and text
    photo_image = ImageTk.PhotoImage(Image.open(image_path).resize((512, 512)))  # Increase the size of the image
    image_label.config(image=photo_image)
    image_label.image = photo_image
    actual_label.config(text=f'\nCorrect: {actual_class if actual_class!="U.K. of Great Britain and Northern Ireland" else "United Kingdom"}')
    predicted_label.config(text=f'Predicted: {class_names[predicted_class] if class_names[predicted_class]!="U.K. of Great Britain and Northern Ireland" else "United Kingdom"}\n')

    # Update the result label
    if actual_class == class_names[predicted_class]:
        result_label.config(text='Correct', bg='green')
    else:
        result_label.config(text='Wrong', bg='red')

    # Recreate map
    map_ax.clear()
    map_ax.set_title('World Map', color="white", size=26)
    map_ax.add_feature(cfeature.COASTLINE)
    map_ax.add_feature(cfeature.BORDERS)
    map_ax.add_feature(cfeature.LAND)
    map_ax.add_feature(cfeature.OCEAN)
    # map_ax.gridlines()
    map_ax.set_global()

    # Show the coordinate on the map
    image_id = os.path.splitext(os.path.basename(image_path))[0][:-4]  # Extract image ID
    if image_id in coordinate_data:
        coordinate = coordinate_data[image_id]
        map_ax.scatter(coordinate[1], coordinate[0], marker='o', color='red', transform=ccrs.PlateCarree(), s=150)
        plt.draw()

def show_next_image():
    global auto_playing, image_index
    image_index = (image_index + 1) % len(image_paths)
    auto_playing = False
    show_image()

def show_previous_image():
    global auto_playing, image_index
    image_index = (image_index - 1) % len(image_paths)
    auto_playing = False
    show_image()

def start_auto_play():
    global auto_playing
    if auto_playing == True: return
    auto_playing = True
    auto_play()

def stop_auto_play():
    global auto_playing
    auto_playing = False

def auto_play():
    global auto_playing, image_index
    if auto_playing == False: return
    image_index = (image_index + 1) % len(image_paths)
    show_image()
    root.after(3500, auto_play)  # Call auto_play function again after 2000 milliseconds (2 seconds)

def quit_app():
    root.destroy()
    sys.exit()

# Create the GUI
root = tk.Tk()

# Set the window size to full screen
root.attributes('-fullscreen', True)

# Set the background color to dark mode
root.configure(background='black')

# Create a frame for everything
main_frame = tk.Frame(root, bg='black')
main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Place the frame on the left side of the window

# Create a frame for the image and text
frame = tk.Frame(main_frame, bg='black')
frame.pack(fill=tk.BOTH, expand=True)  # Place the frame in the top-left corner

# Create the image label
image_label = tk.Label(frame, bg='black')
image_label.pack(fill=tk.BOTH, expand=True)

# Create a frame for the image and text
frame = tk.Frame(main_frame, bg='black')
frame.pack(fill=tk.BOTH, expand=True)  # Place the frame in the top-left corner

# Create a frame for the labels (actual, predicted)
label_frame = tk.Frame(frame, bg='black')
label_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)  # Pack the label frame to the left
label_frame.pack_propagate(False)
label_frame.configure(width=400)
label_frame.configure(height=200)

# Create the actual label
actual_label = tk.Label(label_frame, font=('Arial', 28), fg='white', bg='black')  # Increase the font size
actual_label.pack(fill=tk.X)  # Pack the actual label to fill horizontally


# Create the predicted label
predicted_label = tk.Label(label_frame, font=('Arial', 28), fg='white', bg='black')  # Increase the font size
predicted_label.pack(fill=tk.X)  # Pack the predicted label to fill horizontally

# Create a frame for the result label
result_frame = tk.Frame(frame, bg='black')
result_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)  # Pack the result frame to the right

# Create the result label
result_label = tk.Label(result_frame, font=('Arial', 40), fg='white', width=10, height=5)  # Increase the font size, set a constant size
result_label.pack()  # Pack the result label to fill horizontally

# Create a frame for the buttons
button_frame = tk.Frame(main_frame, bg='black')
button_frame.pack(fill=tk.X)

# Create the next button
next_button = tk.Button(button_frame, text='Next', command=show_next_image, font=('Arial', 26), bg='black', fg='white')  # Increase the font size
next_button.pack(side=tk.LEFT, fill=tk.X, expand=True)  # Place the button in the left side of the button frame

# Create the previous button
previous_button = tk.Button(button_frame, text='Previous', command=show_previous_image, font=('Arial', 26), bg='black', fg='white')  # Increase the font size
previous_button.pack(side=tk.LEFT, fill=tk.X, expand=True)  # Place the button in the left side of the button frame

# Create the auto play button
start_auto_play_button = tk.Button(button_frame, text='Start Auto Play', command=start_auto_play, font=('Arial', 26), bg='black', fg='white')  # Increase the font size
start_auto_play_button.pack(side=tk.LEFT, fill=tk.X, expand=True)  # Place the button in the left side of the button frame

# Create the stop auto play button
stop_auto_play_button = tk.Button(button_frame, text='Stop Auto Play', command=stop_auto_play, font=('Arial', 26), bg='black', fg='white')  # Increase the font size
stop_auto_play_button.pack(side=tk.LEFT, fill=tk.X, expand=True)  # Place the button in the left side of the button frame

# Create the quit button
quit_button = tk.Button(button_frame, text='Quit', command=quit_app, font=('Arial', 26), bg='black', fg='white')  # Increase the font size
quit_button.pack(side=tk.LEFT, fill=tk.X, expand=True)  # Place the button in the left side of the button frame

# Create a frame for the result
result_frame = tk.Frame(root, bg='black')
result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Create a frame for the map
map_frame = tk.Frame(result_frame, bg='black')
map_frame.pack(fill=tk.BOTH, expand=True)

# Create a Cartopy world map
map_fig = plt.figure(figsize=(6, 6), facecolor="black")
map_ax = map_fig.add_subplot(111, projection=ccrs.PlateCarree())
map_fig.tight_layout()

# Create a FigureCanvasTkAgg widget to display the map in Tkinter
canvas = FigureCanvasTkAgg(map_fig, master=map_frame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Show the first image
show_image()

# Start the GUI
root.mainloop()
