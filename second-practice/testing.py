import numpy as np
from keras.api.preprocessing import image
from keras.api.models import load_model

test_image = image.load_img('catsxdogs/single_prediction/chino1.jpg', target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image

model = load_model('./catsxdogs_mobilenet.h5')
predictions = model.predict(test_image)
print(f'Valor da predição: {predictions}')
if predictions[0][0] > 0.5:
    print("É um cachorro")
else:
    print("É um gato")