from keras.models import Sequential
from keras.layers import Dense, Dropout, GaussianNoise, Conv1D, MaxPooling1D, Flatten, BatchNormalization, LeakyReLU, Flatten, MaxPooling2D
from keras import callbacks
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

import math
from keras.preprocessing import image
from keras import backend as K
from keras import regularizers


ADDRESS = 'Put the address to all the images here'
BATCHSIZE = 60
EPOCHCOUNT = 200
IMAGESIZE = 28
VALIDATIONSPLIT = 0.1

def getValset():
    dataset = np.loadtxt('Kannada-MNIST/Dig-MNIST.csv', delimiter=',', skiprows = 1)
    labels = dataset[:, 0] #first column = y
    dataset = dataset[:, 1:] #shape = (#imgs, pixels)

    return dataset, labels

def getTrainset():
    dataset = np.loadtxt('Kannada-MNIST/train.csv', delimiter=',', skiprows = 1)
    labels = dataset[:, 0] #first column = y
    dataset = dataset[:, 1:] #shape = (#imgs, pixels)

    return dataset, labels

def getTestset():
    dataset = np.loadtxt('Kannada-MNIST/test.csv', delimiter=',', skiprows = 1)
    labels = dataset[:, 0] #first column = y
    dataset = dataset[:, 1:] #shape = (#imgs, pixels)

    return dataset, labels
    
def train():
    x, y = getValset()
    xShape = np.shape(x)
    x = np.reshape(x, newshape = (xShape[0], IMAGESIZE, IMAGESIZE)) #reshapes datset to be (#imgs, pixelRow, pixelCol)
    x = tf.data.Dataset.from_tensor_slices(x, y.all())
    x = x.batch(BATCHSIZE)

    mode = stats.mode(y, keepdims = False)
    ybase = np.repeat(mode, xShape[0])
    
    split = (1 - VALIDATIONSPLIT) * x.cardinality().numpy()

    trainSet = x.take(split)
    testSet = x.skip(split)

    return(trainSet, testSet, ybase)

if __name__ == "__main__":
    
    x, y = getTrainset()
    x = np.vstack((x, x))
    y = np.append(y,y)
    print(x.shape)


    model = Sequential()
    model.add(GaussianNoise(0.01, input_dim=x.shape[1]))

    # # model.add(GaussianNoise(0.1))
    # model.add(Dropout(0.2))
    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.5))

    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.5))
    model.add(Flatten())

    model.add(Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(LeakyReLU(alpha=0.5))


    model.add(Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.5))
    model.add(Flatten())

    model.add(Dense(8, kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.5))


    model.add(Dense(4, kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    # model.add(Dense(2, kernel_regularizer=regularizers.l1(0.001), activation='elu'))
    # model.add(Dense(8, kernel_regularizer=regularizers.l1(0.001), activation='elu'))
    # model.add(Dense(50, activation='elu'))
    # model.add(Dense(50, activation='elu'))
    # model.add(Dropout(0.2))

    
    model.add(Dense(1))

    # model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    
    model.summary()

    val_x, val_y = getValset()
    earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=8, restore_best_weights=True)

    # K.set_value(model.optimizer.learning_rate, 1e-3)

    hist = model.fit(x, y, epochs=EPOCHCOUNT, batch_size=BATCHSIZE, shuffle=True, validation_data=(val_x, val_y), callbacks=[earlystopping])
    # hist = model.fit(datagen, epochs=EPOCHCOUNT, batch_size=BATCHSIZE, validation_data=(val_x, val_y), callbacks=[earlystopping])
    
    err = hist.history['mean_squared_error']
    val_err = hist.history['val_mean_squared_error']
    # err = hist.history['mae']
    # val_err = hist.history['val_mae']
    epochs = range(1, len(err) + 1)
    

    print(r2_score(y, model.predict(x)))


    score = np.sqrt(mean_squared_error(y, model.predict(x)))
    print("The Mean Absolute Error of our Model is {}".format(round(score, 2)))


    print(r2_score(val_y, model.predict(val_x)))
    score = np.sqrt(mean_squared_error(val_y, model.predict(val_x)))
    print("The Mean Absolute Error of our Model is {}".format(round(score, 2)))

    plt.plot(epochs, err, '-', label='Training MAE')
    plt.plot(epochs, val_err, ':', label='Validation MAE')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend(loc='upper right')
    plt.plot()
    plt.show()



    