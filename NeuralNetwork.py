import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import csv

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

(trainingX,trainingY), (testX, testY) = mnist.load_data()

# building the input vector from 28x28 pixels
trainingX = trainingX.reshape(60000, 784)
testX = testX.reshape(10000, 784)
trainingX = trainingX.astype('float32')
testX = testX.astype('float32')

digitClasses = 10

#hot encoding
trainingY = np_utils.to_categorical(trainingY, digitClasses)
testY = np_utils.to_categorical(testY, digitClasses)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
#
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
#
model.add(Dense(10))
model.add(Activation('softmax'))

print("Compiling the Model...")
#compiling the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

#training the model.
print("Training the model...")

trainedModel = model.fit(trainingX, trainingY,
          batch_size=128, epochs=20,
          verbose=0,
          validation_data=(testX, testY))

accuracy = model.evaluate(testX, testY, verbose=2)

print("Computing Accuracy...")
print("Test Loss", accuracy[0])
print("Test Accuracy", accuracy[1])

print("Predicting the Digits...")
with open('Final sample.csv') as fin:
 reader = csv.reader(fin)
 for row in reader:
    print('Original Digit is: ', row[0])
    list=row[1:785]
    newList = np.array(list)
    newList=[int(i) for i in list]
    inputDig = np.array([newList])
    print('Predicted Digit is: ',model.predict_classes(inputDig,batch_size=32,verbose=2))
    print('-----------------------------------------------------------')












