# 1 Import Libraries

import keras
import tensorflow.keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.constraints import maxnorm
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD

# 2 import dataset

# Load Dataset and split it into train and test 
from keras.datasets import cirfar10
(x_train, y_train), (x_test, y_test) = cirfar10.load_data()

# Let's check few image of this dataset
# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()


# 3 Preprocess this dataset

# convert class vectors to binary class matrices
from tensorflow.keras.utils import to_categorical
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# convert form integers to floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalize to range 0-1
x_train = x_train / 255
x_test = x_test / 255


# Let's check normalized image
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Model Building

# Building Models
# Adding  layers one by one
# Since this is a classification problem - 
# activation function for hidden layer - relu
# activation function for output layer - softmax
# Input shape (since coloured and 32x32 pixels) - (32,32,3)
# Output shape - 10 
# Drop out layer is added to deactivate some of the neurons for that layer
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3),
                 activation='relu', padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3),
                 activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),
                 activation='relu', padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3),
                 activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3),
                 activation='relu', padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3),
                 activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
print(model.summary())