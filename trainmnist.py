import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # tensorflow 2.0
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('mnist.npz')
print("Train dataset: X: {}, Y: {}".format(x_train.shape, y_train.shape))
print("Test dataset: X: {}, Y: {}".format(x_test.shape, y_test.shape))
n_labels = 10
train_size = y_train.shape[0] ; test_size = y_test.shape[0]
x_train = x_train.reshape(-1, 28, 28, 1)
x_train = np.pad(x_train, [(0,0), (2,2), (2,2), (0,0)], 'constant')
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = np.pad(x_test, [(0,0), (2,2), (2,2), (0,0)], 'constant')
x_norm_train = (x_train - np.mean(x_train, axis=(1,2), keepdims=True))
x_norm_test = (x_test - np.mean(x_test, axis=(1,2), keepdims=True))
y_oh_train = to_categorical(y_train, num_classes=n_labels)
y_oh_test = to_categorical(y_test, num_classes=n_labels)
print("Updated Image Shape: {}".format(x_train[0].shape))
model = keras.Sequential()
model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1,1), activation='relu',input_shape=(32,32,1)))
model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(1,1)))
# C3 layer
model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1,1), activation='relu'))
# S4 layer
model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))
# C5 Fully connected
model.add(layers.Conv2D(filters=120, kernel_size=(5, 5), strides=(1,1), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(units=84, activation='relu'))
model.add(layers.Dense(units=n_labels, activation = 'softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.summary() # Print a summary of the model
# train and keep history
history = model.fit(x_norm_train, y_oh_train, epochs=10, batch_size=64,validation_split=0.2)
# test model
test_res = model.evaluate(x_norm_test, y_oh_test, verbose=0)
print("Test loss: {} , Test accuracy: {}".format(test_res[0], test_res[1]))
# print some of the training history
print("History object records {}".format(history.history.keys()))
print("For example, validation accuracy: {}".format(history.history.get('val_acc')))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['acc'], label='accuracy')