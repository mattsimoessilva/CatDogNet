from PIL import Image
import os

def load_and_preprocess_image(file_path, target_size=(128, 128)):
    """Load and preprocess an image using PIL."""
    try:
        image = Image.open(file_path)
        image = image.resize(target_size)
        # Normalize pixel values without NumPy
        image_data = [pixel / 255.0 for pixel in image.getdata()]
        return image_data
    except IOError:
        print(f"Error opening or processing image at path: {file_path}")
        return None

def load_and_preprocess_images(image_paths, labels):
    """Load and preprocess a list of images with their corresponding labels."""
    images = [load_and_preprocess_image(path) for path in image_paths]
    images = [img for img in images if img is not None]  # Remove None values
    labels = [label for img, label in zip(images, labels) if img is not None]  # Corresponding labels
    return images, labels

def create_labels(image_paths):
    """Create labels for images based on their file names."""
    return [1 if 'cat' in os.path.basename(path) else 0 for path in image_paths]

def split_data(images, labels, train_ratio=0.8):
    """Split data into training and testing sets."""
    num_train = int(len(images) * train_ratio)
    return images[:num_train], labels[:num_train], images[num_train:], labels[num_train:]