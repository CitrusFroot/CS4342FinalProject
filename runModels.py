import numpy as np
import tensorflow as tf
from scipy import stats
from shallow_model.randomForest import *
from deep_model.DeepNN import *
from neural_network_model.three_layer_NN import *

BATCHSIZE = 64
TREEDEPTH = 50
EPOCHCOUNT = 0
IMAGESIZE = 28
VALIDATIONSPLIT = 0.3

#Loads the dataset from the csv files
#returns: dataset, an nxm matrix where n is the number of images, and m is the number of pixels
#returns: labels, a column vector of ground truth values for dataset
def getDataset(address):
    dataset = np.loadtxt(address, delimiter=',', skiprows = 1) #skip header row
    labels = dataset[:, 0] #first column = y
    dataset = dataset[:, 1:] #shape = (#imgs, pixels)

    return dataset, labels

#driver function that runs both models on the data
def train():
    x, y = getDataset('Kannada-MNIST/train.csv')
    x2, y2 = getDataset('Kannada-MNIST/Dig-MNIST.csv')
    print('Datasets loaded.')

    xShape = np.shape(x)
    mode = stats.mode(y, keepdims = False).mode
    ybase = np.repeat(mode, xShape[0])
    print('=========== Baseline Accuracy and loss =============')
    print('baseline accuracy:', fPC(y, ybase))
    print('baseline MSE     :', fMSE(y, ybase))
    print('====================================================\n')

    #============= Shallow Model ===================
    print('Train.csv results')
    split = (1 - VALIDATIONSPLIT) * xShape[0]
    rfModel(x, y, split, TREEDEPTH, x2, y2)

    #============= Deep Model ======================
    deepNN()

    #======== 3-Layer Neural Network Model =========
    three_layer_NN()


#metric functions
def fPC(y, yhat):
    return np.mean(np.equal(y, yhat))

def fMSE(y, yhat):
    return np.mean(np.square(np.subtract(yhat, y)))

train()