
import tensorflow as tf
import numpy as np

from tfHelper import tfHelper

import os
import data
# import matplotlib
import matplotlib.pyplot as plt

from PIL import Image

k = tf.keras

tfHelper.log_level_decrease()
# tfHelper.numpy_show_entire_array(28)
# np.set_printoptions(threshold='nan', linewidth=114)
np.set_printoptions(linewidth=114)

data_augmentation = False
batch_size = 128
num_classes = 10
epochs = 100
imgWidth = 28
# path = './smallmnist/'
path = './new/'
convertColor = 'L'

print ("Load data ...")
# (x_train, y_train), (x_test, y_test) = data.load_data_train()
(x_train, y_train) = tfHelper.get_dataset_with_folder(path, convertColor)
# (x_train, y_train) = tfHelper.get_dataset_with_folder('mnist_png/training/', convertColor)
(x_test, y_test) = tfHelper.get_dataset_with_folder('mnist_png/testing/', convertColor)
# X_pred, X_id, label = data.load_data_predict()
# print(x_train[0])
# exit(0)


print(x_train.shape, 'train samples')
x_train = x_train.reshape(x_train.shape[0], imgWidth, imgWidth, 1)
x_test = x_test.reshape(x_test.shape[0], imgWidth, imgWidth, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
print(x_train[0])

# y_train = k.utils.to_categorical(y_train, num_classes)
# y_test = k.utils.to_categorical(y_test, num_classes)

model = k.models.Sequential()
model.add(k.layers.Conv2D(64, (3, 3), padding='same',
                 input_shape=(imgWidth,imgWidth,1)))
model.add(k.layers.Activation('relu'))
model.add(k.layers.Conv2D(64, (3, 3)))
model.add(k.layers.Activation('relu'))
model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(k.layers.Dropout(0.25))


model.add(k.layers.Flatten())
model.add(k.layers.Dense(512))
model.add(k.layers.Activation('relu'))
# model.add(k.layers.Dropout(0.1))
model.add(k.layers.Dense(80))
model.add(k.layers.Activation('relu'))
model.add(k.layers.Dropout(0.5))
model.add(k.layers.Dense(num_classes))
model.add(k.layers.Activation('softmax'))
# initiate RMSprop optimizer
opt = k.optimizers.Adam(lr=1e-04)
# opt = k.optimizers.Adam(lr=0.0001, decay=1e-6)
# opt = k.optimizers.rmsprop(lr=0.0001, decay=1e-6)

learning_rate_reduction = k.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                        patience=1, 
                                                        verbose=1, 
                                                        factor=0.5, 
                                                        min_lr=1e-09)

tensorBoard = k.callbacks.TensorBoard()

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=100,
          # validation_data=(x_train, y_train),
          validation_data=(x_test, y_test),
          shuffle=True,
          callbacks=[learning_rate_reduction, tensorBoard]
          )
tfHelper.save_model(model, "model")
