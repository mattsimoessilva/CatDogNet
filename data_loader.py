from image_processing import load_and_preprocess_images, create_labels, split_data
import os
import cv2

def check_and_resize_image(image_path, target_dimensions):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image

def check_image_dimensions(image_paths, target_dimensions=(224, 224)):
    # Check dimensions of the first image
    first_image = cv2.imread(image_paths[0])
    first_dimensions = first_image.shape

    # Check dimensions of other images
    for path in image_paths[1:]:
        image = cv2.imread(path)
        if image.shape != first_dimensions:
            print(f"Resizing image: {path}")
            resized_image = check_and_resize_image(path, target_dimensions)
            cv2.imwrite(path, resized_image)

def load_data(data_path):
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
        image = image.reshape(-1) # Flatten the image
        train_data.append(image)
        train_labels.append(label)

    # Load and preprocess test images
    for file_name in os.listdir(test_path):
        file_path = os.path.join(test_path, file_name)

        # Skip non-image files
        if not file_path.endswith(('.jpg', '.png', '.jpeg')):
            continue

        # Load and preprocess test images
        image = cv2.imread(file_path)
        image = image.reshape(-1) # Flatten the image
        test_data.append(image)

    return train_data, train_labels, test_data, test_labels
