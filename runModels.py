import numpy as np
import tensorflow as tf
from scipy import stats
from shallow_model.randomForest import *
#from deep_model.neuralNetwork import *

BATCHSIZE = 64
TREEDEPTH = 45
EPOCHCOUNT = 0
IMAGESIZE = 28
VALIDATIONSPLIT = 0.3

#Loads the dataset from the csv files
#returns: dataset, an nxm matrix where n is the number of images, and m is the number of pixels
#returns: labels, a column vector of ground truth values for dataset
def getDataset():
    dataset = np.loadtxt('Kannada-MNIST/train.csv', delimiter=',', skiprows = 1) #skip header row
    labels = dataset[:, 0] #first column = y
    dataset = dataset[:, 1:] #shape = (#imgs, pixels)

    return dataset, labels

#driver function that runs both models on the data
def train():
    x, y = getDataset()
    print('Dataset loaded.')

    xShape = np.shape(x)
    mode = stats.mode(y, keepdims = False).mode
    print(mode)
    ybase = np.repeat(mode, xShape[0])
    print('=========== Baseline Accuracy and loss =============')
    print('baseline accuracy:', fPC(y, ybase))
    print('baseline MSE     :', fMSE(y, ybase))
    print('====================================================\n')

    
    #============= Shallow Model ===================
    split = (1 - VALIDATIONSPLIT) * xShape[0]
    rfModel(x, y, split, TREEDEPTH)

    #============= Deep Model ======================
    x = np.reshape(x, newshape = (xShape[0], IMAGESIZE, IMAGESIZE)) #reshapes datset to be (#imgs, pixelRow, pixelCol)
    x = tf.data.Dataset.from_tensor_slices(x, y.all())

    neuralNet(x, y, EPOCHCOUNT, BATCHSIZE, VALIDATIONSPLIT)

#metric functions
def fPC(y, yhat):
    return np.mean(np.equal(y, yhat))

def fMSE(y, yhat):
    return np.mean(np.square(np.subtract(yhat, y)))

train()