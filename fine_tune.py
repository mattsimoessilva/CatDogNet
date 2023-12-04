from data_loader import image_generator, get_files
import tensorflow as tf
import numpy as np

# Load and preprocess the specific image
img = tf.keras.preprocessing.image.load_img('teste2.jpg', target_size=(64, 64))
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
label = np.array([[1, 0]])  # assuming 'cat' is represented by [1, 0]

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Fine-tune the model on the specific image
model.fit(img, label, epochs=10)

# Load the test dataset
test_files = get_files('test/test')
test_generator = image_generator(test_files, batch_size=32)
X_test, y_test = next(test_generator)

# Continue training on the test dataset
model.fit(X_test, y_test, epochs=10)

# Save the fine-tuned model
model.save('fine_tuned_model.h5')

