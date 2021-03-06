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

print(conv_base.summary())
base_dir = "D:/dataset/ds images/kaggle/"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 5 , 5, 1280))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory=directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 5 * 5* 1280))
validation_features = np.reshape(validation_features, (1000,5 * 5* 1280))
test_features = np.reshape(test_features, (1000, 5 * 5* 1280))

model = Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=5 * 5* 1280))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(
    train_features,
    train_labels,
    epochs=30,
    verbose=1,
    batch_size=20,
    validation_data=(validation_features, validation_labels)
)

model.save('cats_and_dog_MbV2.h5')

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
