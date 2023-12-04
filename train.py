import tensorflow as tf
from sklearn.utils import class_weight
from data_loader import image_generator, finite_image_generator, get_files
from model import create_model
import numpy as np

# Get the file paths
train_files = get_files('train')
test_files = get_files('test')

# Create the finite image generator for computing class weights
train_generator_finite = finite_image_generator(train_files, batch_size=32)

# Compute class weights
y_train = np.array([pair[1] for pair in train_generator_finite])
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

# Create the infinite image generators for training
train_generator = image_generator(train_files, batch_size=32)
test_generator = image_generator(test_files, batch_size=32)

# Create an ImageDataGenerator object for data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# Create the model
model = create_model()

# Train the model
steps_per_epoch = len(train_files) / 32
validation_steps = len(test_files) / 32
model.fit(datagen.flow(train_generator, batch_size=32),
          steps_per_epoch=steps_per_epoch,
          epochs=100,
          class_weight=class_weights,
          validation_data=test_generator,
          validation_steps=validation_steps)

# Save the model
model.save('model.h5')


