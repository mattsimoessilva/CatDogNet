from image_processing import load_and_preprocess_images, create_labels, split_data
import os
import cv2

def load_data(data_path):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # Assuming that the 'train' and 'test' folders contain the images
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')

    cat_images = []
    dog_images = []

    # Load training data
    for file_name in sorted(os.listdir(train_path)):
        file_path = os.path.join(train_path, file_name)

        # Skip non-image files
        if not file_path.endswith(('.jpg', '.png', '.jpeg')):
            continue

        # Load and preprocess training images
        image = cv2.imread(file_path)
        image = image.reshape(-1) # Flatten the image

        if 'cat' in file_name:
            cat_images.append((image, 1))  # 1 for cat
        elif 'dog' in file_name:
            dog_images.append((image, 0))  # 0 for dog

    # Load only half of the cats and dogs
    half_cat = len(cat_images) // 10
    half_dog = len(dog_images) // 10

    train_data += [img for img, label in cat_images[:half_cat]]
    train_labels += [label for img, label in cat_images[:half_cat]]

    train_data += [img for img, label in dog_images[:half_dog]]
    train_labels += [label for img, label in dog_images[:half_dog]]

    # Load and preprocess test images
    for file_name in sorted(os.listdir(test_path)):
        file_path = os.path.join(test_path, file_name)

        # Skip non-image files
        if not file_path.endswith(('.jpg', '.png', '.jpeg')):
            continue

        # Load and preprocess test images
        image = cv2.imread(file_path)
        image = image.reshape(-1) # Flatten the image
        test_data.append(image)

    return train_data, train_labels, test_data, test_labels
