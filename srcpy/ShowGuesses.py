import tkinter as tk
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import random

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

def show_image():
    global image_index

    # Load the image
    image = load_img(image_paths[image_index], target_size=(128, 128))
    image = img_to_array(image).reshape((1, 128, 128, 3)) / 255.0

    # Make a prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Get the actual class
    actual_class = os.path.basename(os.path.dirname(image_paths[image_index]))

    # Update the image and text
    photo_image = ImageTk.PhotoImage(Image.open(image_paths[image_index]).resize((512, 512)))  # Increase the size of the image
    image_label.config(image=photo_image)
    image_label.image = photo_image
    actual_label.config(text=f'Actual: {actual_class if actual_class!="U.K. of Great Britain and Northern Ireland" else "United Kingdom"}')
    predicted_label.config(text=f'Predicted: {class_names[predicted_class] if class_names[predicted_class]!="U.K. of Great Britain and Northern Ireland" else "United Kingdom"}')

    # Update the result label
    if actual_class == class_names[predicted_class]:
        result_label.config(text='Correct', bg='green')
    else:
        result_label.config(text='Wrong', bg='red')

def show_next_image():
    global image_index
    image_index = (image_index + 1) % len(image_paths)
    show_image()

def show_previous_image():
    global image_index
    image_index = (image_index - 1) % len(image_paths)
    show_image()

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

# Create the actual label
actual_label = tk.Label(frame, font=('Arial', 24), fg='white', bg='black')  # Increase the font size
actual_label.pack(fill=tk.X)

# Create the predicted label
predicted_label = tk.Label(frame, font=('Arial', 24), fg='white', bg='black')  # Increase the font size
predicted_label.pack(fill=tk.X)

# Create a frame for the buttons
button_frame = tk.Frame(main_frame, bg='black')
button_frame.pack(fill=tk.X)

# Create the next button
next_button = tk.Button(button_frame, text='Next', command=show_next_image, font=('Arial', 24), bg='black', fg='white')  # Increase the font size
next_button.pack(side=tk.LEFT, fill=tk.X, expand=True)  # Place the button in the left side of the button frame

# Create the previous button
previous_button = tk.Button(button_frame, text='Previous', command=show_previous_image, font=('Arial', 24), bg='black', fg='white')  # Increase the font size
previous_button.pack(side=tk.LEFT, fill=tk.X, expand=True)  # Place the button in the left side of the button frame

# Create a frame for the result
result_frame = tk.Frame(root, bg='black')
result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Create the result label
result_label = tk.Label(result_frame, font=('Arial', 24), fg='white', width=10, height=5)  # Increase the font size, set a constant size
result_label.pack()

# Show the first image
show_image()

# Start the GUI
root.mainloop()
