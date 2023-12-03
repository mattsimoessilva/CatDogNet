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
    for file_name in os.listdir(train_path):
        file_path = os.path.join(train_path, file_name)

        # Skip non-image files
        if not file_path.endswith(('.jpg', '.png', '.jpeg')):
            continue

        label = 1 if 'cat' in file_name else 0  # 1 for cat, 0 for dog

        # Load and preprocess training images
        train_data.append(file_path)
        train_labels.append(label)

        # Load and preprocess test images
        test_data = [os.path.join(test_path, file_name) for file_name in os.listdir(test_path) if file_name.endswith(('.jpg', '.png', '.jpeg'))]

        return train_data, train_labels, test_data, test_labels
