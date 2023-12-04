import os
import cv2
import numpy as np

def load_data(files):
    data = []
    labels = []
    for file in files:
        # Read the image
        img = cv2.imread(file)
        # Resize the image
        img = cv2.resize(img, (64, 64))
        # Normalize the image
        img = img / 255.0
        # Append the image to the data list
        data.append(img)
        # Append the label to the labels list
        label = [1, 0] if 'cat' in file else [0, 1]
        labels.append(label)
    # Convert the lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def get_files(path):
    return [os.path.join(path, file) for file in os.listdir(path) if file.endswith(('.jpg', '.png', '.jpeg'))]

def image_generator(files, batch_size=32):
    while True:
        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a=files, size=batch_size)
        batch_input, batch_output = load_data(batch_paths)
        yield (batch_input, batch_output)
