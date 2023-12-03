import os

def load_data(data_path):
    train_data = []
    train_labels = []
    test_data = []

    # Assuming that the 'train' and 'test' folders contain the images
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')

    # Load training data
    for category in os.listdir(train_path):
        category_path = os.path.join(train_path, category)

        # Skip non-directory entries
        if not os.path.isdir(category_path):
            continue

        label = [1, 0] if category == 'cat' else [0, 1]  # [1, 0] for cat, [0, 1] for dog

        for file_name in os.listdir(category_path):
            image_path = os.path.join(category_path, file_name)
            train_data.append(image_path)
            train_labels.append(label)

    # Load test data
    for file_name in os.listdir(test_path):
        image_path = os.path.join(test_path, file_name)
        test_data.append(image_path)

    return train_data, train_labels, test_data
