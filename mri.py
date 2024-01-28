import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model('mri_final.keras')

# Define class labels
class_labels = {0: 'glioma', 1: 'meningioma', 2: 'no tumor', 3: 'pituitary'}

# Function to preprocess the image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(200, 200))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    return img_array

# Function to classify the image
def classify_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    class_label = class_labels[class_index]
    return class_label, predictions[0, class_index]

# Function to handle image selection
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        class_label, confidence = classify_image(file_path)
        result_label.config(text=f'Class: {class_label}\nConfidence: {confidence:.4f}')
        image = Image.open(file_path)
        image = image.resize((200, 200))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
    else:
        messagebox.showerror("Error", "No file selected")

# Create the main application window
root = tk.Tk()
root.title("MRI Image Classifier")

# Create widgets
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
