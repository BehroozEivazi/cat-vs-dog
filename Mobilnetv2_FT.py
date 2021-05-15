from keras.applications import MobileNetV2
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow
from keras.utils import to_categorical
from keras.models import Sequential
import os
import numpy as np

conv_base = MobileNetV2(weights='imagenet',
                        include_top=False,
                        input_shape=(150, 150, 3)
                        )

base_dir = "D:/dataset/ds images/kaggle/"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20

conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == "block_16_project":
        set_trainable = True
    if set_trainable == True:
        layer.trainable = True
    else:
        layer.trainable = False
print(conv_base.summary())

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)
test_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

model = Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
print(model.summary())
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=5,
    verbose=1,
    validation_data=test_generator,
    shuffle=True,
    validation_steps=50
)

model.save('cats_and_dog_MbV2FT.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
