import numpy as np
import tensorflow as tf
import cv2
import os
from keras.api.utils import to_categorical
from keras.api.models import Sequential
from keras.api.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping
import matplotlib.pyplot as plt

trainDataPath = "train"
testDataPath = "validation"

X_train = []
Y_train = []

folderList = os.listdir(trainDataPath)
folderList.sort()

#print(folderList)
#load data
print("Start data load...")
for i, category in enumerate(folderList):
    files = os.listdir(trainDataPath+'/'+category)
    for file in files:
        #print(category+'/'+file)
        img = cv2.imread(trainDataPath+"/"+category+'/{0}'.format(file),0)
        X_train.append(img)
        Y_train.append(i)
print("Finish data load")

print("Train data:")
print(len(X_train))
#print(Y_train)
print(len(Y_train))


X_test = []
Y_test = []

folderList = os.listdir(testDataPath)
folderList.sort()

#load test data

print("Start data load...")
for i, category in enumerate(folderList):
    files = os.listdir(testDataPath+'/'+category)
    for file in files:
        #print(category+'/'+file)
        img = cv2.imread(testDataPath+"/"+category+'/{0}'.format(file),0)
        X_test.append(img)
        Y_test.append(i)
print("Finish data load")

print("Test data:")
print(len(X_test))
#print(Y_test)
print(len(Y_test))

#numpy

X_train = np.array(X_train, 'float32')
Y_train = np.array(Y_train, 'float32')
X_test = np.array(X_test, 'float32')
Y_test = np.array(Y_test, 'float32')

#normalize
X_train = X_train/255.0
X_test = X_test/255.0

#reshape the image

TrainImageNumber = X_train.shape[0]
X_train = X_train.reshape(TrainImageNumber, 48, 48, 1)


#reshape the image

TestImageNumber = X_test.shape[0]
X_test = X_test.reshape(TestImageNumber, 48, 48, 1)

Y_train = to_categorical(Y_train, num_classes=7)
Y_test = to_categorical(Y_test, num_classes=7)


#Build model

input_shape = X_train.shape[1:]


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs available: {gpus}")
    try:
        for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e) 
else:
    print("No GPUs detected.")



model = Sequential()
model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dense(7,activation="softmax"))

print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

batch=32
epochs=30

stepsPerEpoch = np.ceil(len(X_train)/batch)
validationSteps = np.ceil(len(X_test)/batch)

stopEarly = EarlyStopping(monitor='val_accuracy' , patience=5)

history = model.fit(X_train,
                    Y_train,
                    batch_size=batch,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test,Y_test),
                    shuffle=True,
                    callbacks=[stopEarly])

#plot
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc , 'r' , label="Train accuracy")
plt.plot(epochs, val_acc , 'b' , label="Validation accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Trainig and validation Accuracy")
plt.legend(loc='lower right')
plt.show()


plt.plot(epochs, loss , 'r' , label="Train loss")
plt.plot(epochs, val_loss , 'b' , label="Validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Trainig and validation Loss")
plt.legend(loc='upper right')
plt.show()


modelFileName = "emotion.h5"
model.save(modelFileName)