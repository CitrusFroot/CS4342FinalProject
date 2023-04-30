from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import math


ADDRESS = 'Put the address to all the images here'
BATCHSIZE = 150
EPOCHCOUNT = 20
IMAGESIZE = 28
VALIDATIONSPLIT = 0.1

def getDataset():
    dataset = np.loadtxt('Kannada-MNIST/Dig-MNIST.csv', delimiter=',', skiprows = 1)
    labels = dataset[:, 0] #first column = y
    dataset = dataset[:, 1:] #shape = (#imgs, pixels)

    return dataset, labels

def getTestset():
    dataset = np.loadtxt('Kannada-MNIST/train.csv', delimiter=',', skiprows = 1)
    labels = dataset[:, 0] #first column = y
    dataset = dataset[:, 1:] #shape = (#imgs, pixels)

    return dataset, labels
    
def train():
    x, y = getDataset()
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
    
    x, y = getTestset()

    print(x.shape[1])
    model = Sequential()
    model.add(Dense(100, activation='elu', input_dim=x.shape[1]))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(100, activation='elu'))
    # model.add(Dense(50, activation='elu'))
    # model.add(Dense(50, activation='elu'))
    # model.add(Dense(50, activation='elu'))

    
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mae', metrics=['mae'])#, epsilon='1e-5')
    # model.compile(optimizer='adamax', loss='mean_squared_error', metrics=['mean_squared_error'])
    
    model.summary()

    val_x, val_y = getDataset()
    # hist = model.fit(x, y, epochs=EPOCHCOUNT, batch_size=BATCHSIZE, validation_split=VALIDATIONSPLIT)

    earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)
    hist = model.fit(x, y, epochs=EPOCHCOUNT, batch_size=BATCHSIZE, validation_data=(val_x, val_y), callbacks=[earlystopping])
    
    # err = hist.history['mean_squared_error']
    # val_err = hist.history['val_mean_squared_error']
    err = hist.history['mae']
    val_err = hist.history['val_mae']
    epochs = range(1, len(err) + 1)
    

    print(r2_score(y, model.predict(x)))

    # xTest, yTest = getDataset()
    # print(r2_score(yTest, model.predict(xTest)))
    xTest, yTest = getDataset()
    print(r2_score(val_y, model.predict(val_x)))
    plt.plot(epochs, err, '-', label='Training MAE')
    plt.plot(epochs, val_err, ':', label='Validation MAE')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend(loc='upper right')
    plt.plot()
    plt.show()

    

    # MAE - 93%
    # MSE - 81.3%


    