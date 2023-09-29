from PIL import Image

def load_image(file_path):
    """
    Function to load an image from a file.
    """
    return Image.open(file_path)

def resize_image(image, size):
    """
    Function to resize an image to a desired size.
    """
    return image.resize(size)

def normalize_image(image):
    """
    Function to normalize the pixel values of an image.
    """
    return [pixel / 255.0 for pixel in image.getdata()]

def create_labels(image_paths):
    """
    Function to create labels for images based on their file names.
    """
    labels = []
    for path in image_paths:
        if 'cat' in path:
            labels.append([1, 0])  # [1, 0] for cat
        elif 'dog' in path:
            labels.append([0, 1])  # [0, 1] for dog
    return labels

def split_data(images, labels, train_ratio=0.8):
    """
    Function to split data into training and testing sets.
    """
    num_train = int(len(images) * train_ratio)
    return images[:num_train], labels[:num_train], images[num_train:], labels[num_train:]
