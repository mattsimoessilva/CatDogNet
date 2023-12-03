from image_processing import load_and_preprocess_images, create_labels, split_data
import os

def load_data(data_path):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # Assuming that the 'train' and 'test' folders contain the images
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')

    # Load training data
    for category in os.listdir(train_path):
        category_path = os.path.join(train_path, category)

        # Skip non-directory entries
        if not os.path.isdir(category_path):
            continue

        label = 1 if category == 'cat' else 0  # 1 for cat, 0 for dog

        # Load and preprocess training images
        images = [os.path.join(category_path, file_name) for file_name in os.listdir(category_path)]
        labels = [label] * len(images)
        train_data.extend(images)
        train_labels.extend(labels)

    # Load and preprocess test images
    test_data = [os.path.join(test_path, file_name) for file_name in os.listdir(test_path)]

    return train_data, train_labels, test_data, test_labels
