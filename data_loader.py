import os
from image_processing import load_and_preprocess_images

def load_data(data_path):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # Supondo que as pastas 'train' e 'test' contêm as imagens
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')

    # Carregar dados de treinamento
    for category in os.listdir(train_path):
        category_path = os.path.join(train_path, category)
        label = 1 if 'cat' in category else 0  # 1 para gato, 0 para cachorro

        image_paths = [os.path.join(category_path, file_name) for file_name in os.listdir(category_path)]
        images, labels = load_and_preprocess_images(image_paths, [label] * len(image_paths))

        train_data.extend(images)
        train_labels.extend(labels)

    # Carregar dados de teste
    for category in os.listdir(test_path):
        category_path = os.path.join(test_path, category)
        label = 1 if 'cat' in category else 0  # 1 para gato, 0 para cachorro

        image_paths = [os.path.join(category_path, file_name) for file_name in os.listdir(category_path)]
        images, labels = load_and_preprocess_images(image_paths, [label] * len(image_paths))

        test_data.extend(images)
        test_labels.extend(labels)

    return train_data, train_labels, test_data, test_labels

