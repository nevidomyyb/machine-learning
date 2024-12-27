import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense
import time
from pathlib import Path

base_dir = Path(__file__).parent


TESTSET_DIR = base_dir/ "testset"
TRAINSET_DIR = base_dir/"trainset"

init_time = time.time()

base_model_inceptionv3 = keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
x = base_model_inceptionv3.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)
model = Model(inputs=base_model_inceptionv3.input, outputs=predictions)

for layer in base_model_inceptionv3.layers:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss="categorical_crossentropy", metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    base_dir/"trainset",
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    base_dir/"testset",
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical"
)

for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history_fine_tuning = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10 
)

model.save("geometry-tl.h5")

end_time = time.time()
print(f"The training lasted: {end_time-init_time} seconds")
