import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import os
import tensorflow as tf

TESTSET_DIR = "./testset/"
TRAINSET_DIR = "./trainset/"
BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAINSET_DIR,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names

image_path = "./output/Circle_8a3b4bc8-2a8d-11ea-8123-8363a7ec19e6.png"

img = tf.keras.utils.load_img(image_path, target_size=(150, 150))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

model = load_model('./geometry-tl.h5')


predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100*np.max(score):.2f}% de confiança")

