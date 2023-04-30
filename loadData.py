import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from shallow_model.randomForest import *


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
    split = (1 - VALIDATIONSPLIT) * xShape[0]
    yhatShallow = rfModel(x, y, split)
    x = np.reshape(x, newshape = (xShape[0], IMAGESIZE, IMAGESIZE)) #reshapes datset to be (#imgs, pixelRow, pixelCol)
    x = tf.data.Dataset.from_tensor_slices(x, y.all())

    mode = stats.mode(y, keepdims = False)
    ybase = np.repeat(mode, xShape[0])
    

    trainSet = x.take(split).batch(BATCHSIZE)
    testSet = x.skip(split).batch(BATCHSIZE)

    return(trainSet, testSet, ybase)

train()