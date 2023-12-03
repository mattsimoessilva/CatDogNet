import os
import cv2
import numpy as np

def check_and_resize_image(image_path, target_dimensions):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image

def flatten_image(image):
    return image.reshape(-1)  # Flatten the image

def load_data(data_path, use_percent=0.5):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # Assuming that the 'train' and 'test' folders contain the images
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')

    # Load training data
    for file_name in os.listdir(train_path):
        file_path = os.path.join(train_path, file_name)

        # Skip non-image files
        if not file_path.endswith(('.jpg', '.png', '.jpeg')):
            continue

        label = 1 if 'cat' in file_name else 0  # 1 for cat, 0 for dog

        # Load and preprocess training images
        image = cv2.imread(file_path)
        flat_image = flatten_image(image)
        train_data.append(flat_image)
        train_labels.append(label)

    # Determine the number of samples to use based on the percentage
    num_samples_to_use = int(len(train_data) * use_percent)

    # Use only a subset of the training data
    train_data = train_data[:num_samples_to_use]
    train_labels = train_labels[:num_samples_to_use]

    # Load and preprocess test images
    for file_name in os.listdir(test_path):
        file_path = os.path.join(test_path, file_name)

        # Skip non-image files
        if not file_path.endswith(('.jpg', '.png', '.jpeg')):
            continue

        # Load and preprocess test images
        image = cv2.imread(file_path)
        flat_image = flatten_image(image)
        test_data.append(flat_image)

    return train_data, train_labels, test_data, test_labels
