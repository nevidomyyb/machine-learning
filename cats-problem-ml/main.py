import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import keras
from keras.src.layers import Dense, GlobalAveragePooling2D
from keras.src.applications.mobilenet import MobileNet, preprocess_input
from keras.src.utils import image_dataset_from_directory
from keras.src.models import Model
from keras.src.optimizers import Adam
from keras.src.layers import Dropout

model = MobileNet(weights='imagenet', include_top=False) #Importando o modelo MobileNet, usando os pesos do treinamento com o banco de dados da ImageNet e descartando a última camada de neurônios
#Criando a saída do modelo MobileNet
x = model.output 
x = GlobalAveragePooling2D()(x)

#Adicionando uma camada intermediária e uma camada final
x = Dense(50, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)
model = Model(inputs=model.input, outputs=preds)

#Visualizando todas as camadas da nova rede criada usando o modelo MobileNetV2
for i, layer in enumerate(model.layers):
    print(i, layer.name)
    if i < 88:
        layer.trainable = False
    if i >= 88:
        layer.trainable = True
        
batch_size = 32
img_height = 224
img_width = 224
batch_size = 32

# Carregar e preparar os datasets
train_dataset = image_dataset_from_directory(
    'catsxdogs/training_set',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'
)

test_dataset = image_dataset_from_directory(
    'catsxdogs/test_set',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'
)

# Aumentação de dados usando tf.data
data_augmentation = keras.Sequential([
    keras.layers.Rescaling(1./255),
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomTranslation(0.3, 0.3)
])

# Aplicar a transformação de aumento de dados ao dataset de treinamento
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
histry = model.fit(train_dataset, steps_per_epoch=8000//batch_size, epochs = 10, validation_data=test_dataset, validation_steps=2000//batch_size)
model.save('catsxdogs_mobilenet.h5')
