import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.vgg16 import decode_predictions

# load the VGG16 model with pre-trained ImageNet weights
model = VGG16(weights='imagenet', include_top=True)

# load the image
img_path = 'teste2.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# predict the probability across all output classes
y_prob = model.predict(x) 
y_classes = y_prob.argmax(axis=-1)

print(y_classes)

# convert the probabilities to class labels
label = decode_predictions(y_prob)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))


