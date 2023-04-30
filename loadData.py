import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats


ADDRESS = 'Put the address to all the images here'
BATCHSIZE = 64
EPOCHCOUNT = 0
IMAGESIZE = 28
VALIDATIONSPLIT = 0.3

def getDataset():
    dataset = np.loadtxt('Kannada-MNIST\Dig-MNIST.csv', delimiter=',', skiprows = 1)
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
