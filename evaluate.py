import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import load_model

# load the trained model
model = load_model('model.h5')

# load the image
img_path = 'teste5.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# predict the probability across all output classes
y_prob = model.predict(x) 
y_classes = y_prob.argmax(axis=-1)

# print the class label
if y_classes[0] == 0:
    print('cat')
else:
    print('dog')