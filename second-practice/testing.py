import numpy as np
from keras.api.preprocessing import image
from keras.api.models import load_model
import os

test_images = []
base_dir = "./catsxdogs/single_prediction/"
images = [f"{base_dir}{f}" for f in os.listdir('./catsxdogs/single_prediction/') if os.path.isfile(f"{base_dir}{f}")]

for imagem in images:
    test_image = image.load_img(imagem, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image/255.0
    test_images.append({"name": imagem, "image": test_image})

model = load_model('./catsxdogs_mobilenet.h5')
final_list = []
for test_image in test_images:
    prediction = model.predict(test_image["image"])
    print(test_image['name'].replace(base_dir, ""))
    final_list.append([test_image['name'].replace(base_dir, ""), prediction[0][0]])
    # print(f"Imagem: {test_image['name']}")
    # print(f'Valor da predição: {prediction}')
    # if prediction[0][0] > 0.5:
    #     print("É um cachorro")
    # else:
    #     print("É um gato")
    # print("")
print(final_list)