#%%
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# defining some parameters for training
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 10

# the data, split between test and train data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Shape of Training dataset: " + str(x_train.shape))
print("Shape of Testing dataset: " + str(x_test.shape))

#%%

# reshaping the dataset so it fits keras requirements
print('reshaping...')
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# converting class vecotrs to binary class matrices
print('converting to binary...')
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print("Shape of Training dataset: " + str(x_train.shape))
print("Shape of Testing dataset: " + str(x_test.shape))

# %%

# creating the Convolutional Neural Network (CNN)
# Dropout layers to reduce overfitting
print('creating the model...')
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Compiling the Model using the Adadelta optimizer
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

# %%

# Training the model
print('starting the training process...')
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(x_test, y_test))
print("The model has successfully trained")

# Storing it to disk
print("Saving the model as mnist.h5")
model.save('mnist.h5')
print("The model has been successfully saved")

# %%

# Evaluating the model
print('starting model evaluation...')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
