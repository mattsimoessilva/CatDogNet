import tensorflow as tf
import numpy as np

# Load the image
img = tf.keras.preprocessing.image.load_img('teste3.jpeg', target_size=(64, 64))
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)

# Load the model
model = tf.keras.models.load_model('model.h5')

# Evaluate the model
predictions = model.predict(img)

# Print the prediction
if np.argmax(predictions) == 0:
    print('The model predicts this image is a cat.')
else:
    print('The model predicts this image is a dog.')

