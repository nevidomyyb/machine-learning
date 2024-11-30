import numpy as np
from keras.models import load_model
import os
import tensorflow as tf

TESTSET_DIR = "./testset/"
TRAINSET_DIR = "./trainset/"
BATCH_SIZE = 32
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAINSET_DIR,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
model = load_model('./geometry-tl.h5')

def test():
    all_files = [f for f in os.listdir("./output/") if os.path.isfile(os.path.join("./output/", f))]
    squares = [f for f in all_files if "Square" in f]
    pentagons = [f for f in all_files if "Pentagon" in f]
    hexagons = [f for f in all_files if "Hexagon" in f]
    heptagons = [f for f in all_files if "Heptagon" in f]
    octagons = [f for f in all_files if "Octagon" in f]
    nonagons = [f for f in all_files if "Nonagon" in f]
    circles = [f for f in all_files if "Circle" in f]
    stars = [f for f in all_files if "Star" in f]
    files = [squares, pentagons, hexagons, heptagons, octagons, nonagons, circles, stars]
    for files_type in files:
        i = 0
        acertos = 0
        total = 0
        for i, file in enumerate(files_type):
            if i == 200:
                print(".", end="\n")
                break
            else:
                print(".", end="")
            image_type = file[0:file.find("_")]
            full_path = os.path.join("./output/", file)
            img = tf.keras.utils.load_img(full_path, target_size=(150, 150))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array /= 255.0
            predictions = model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0])
            type_by_model = class_names[np.argmax(score)]

            total+=1
            if type_by_model.lower() == image_type.lower():
                acertos+=1
                        
        print(f"{image_type}: {(acertos/total)*100:.0f}% of accuracy.")
            

test()