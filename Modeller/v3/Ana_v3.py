"""
NOT : GRAPHVIZ ILE ILGILI SORUN YASANIR ISE model_plot ISLEMINDEN VAZGECIN !
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import sys
# %%
num_features = 32
width, height = 48, 48
num_labels = 7
batch_size = 32
epochs = int(sys.argv[1])
learningRate = 0.0001
#
#desinging the CNN

# YENI MODEL
model = Sequential()

model.add(Conv2D(num_features, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(num_features * 2, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(num_features*2*2, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(num_features*2*2, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

plot_model(model,'model_v3.png')

model.summary()

# training the model with loading the numpy data (x-y)

# %%
x_train = np.load('../../numpyDatas/x_train.npy')
y_train = np.load('../../numpyDatas/y_train.npy')

x_valid = np.load('../../numpyDatas/x_valid.npy')
y_valid = np.load('../../numpyDatas/y_valid.npy')
# %%

log_dir = "..\\..\\logs\\fit\\v3"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# %% DATA GENERATOR
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
train_datagen.fit(x_train)
# fits the model on batches with real-time data augmentation:

# Compiling the model with adam optimizer and categorical crossentropy loss
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=learningRate, decay=1e-6),
              metrics=['accuracy'])

history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_datagen.flow(x_valid, y_valid, batch_size=batch_size),
                    shuffle=True,
                    callbacks=[tensorboard_callback])

# %%


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./egitimGrafigi_v3.png')
plt.show()

model.save("model_v3")
model.save_weights("model_weights_v3.h5")
