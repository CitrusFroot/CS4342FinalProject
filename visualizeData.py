import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def getSamples():
    # Load the data
    dataset = np.loadtxt('Kannada-MNIST/Dig-MNIST.csv', delimiter=',', skiprows=1)
    labels = dataset[:, 0]  # first column = y
    dataset = dataset[:, 1:]  # shape = (#imgs, pixels)

    # Reshape the data
    dataset = np.reshape(dataset, (-1, 28, 28))

    # Define the labels
    label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Plot some examples
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(dataset[i], cmap='gray')
        ax.set_title(label_names[int(labels[i])])
        ax.axis('off')
    plt.show()

def get2DPCA():
    data = pd.read_csv('Kannada-MNIST/train.csv')

    X = data.drop('label', axis=1)
    y = data['label']

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)

    principalDF = pd.DataFrame(data = principalComponents, columns = ['pc 1', 'pc 2'])

    finalDF = pd.concat([principalDF, y], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('pc 1', fontsize = 15)
    ax.set_ylabel('pc 2', fontsize = 15)

    targets = list(set(y))
    colors = ['#e6194b', '#f58231', '#ffe119', '#bfef45', '#3cb44b', '#42d4f4', '#4363d8', '#911eb4', '#f032e6', '#a9a9a9']
    for target, color in zip(targets,colors):
        keep = finalDF['label'] == target
        ax.scatter(finalDF.loc[keep, 'pc 1']
                , finalDF.loc[keep, 'pc 2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()

def get3DPCA():
    data = pd.read_csv('Kannada-MNIST/train.csv')

    X = data.drop('label', axis=1)
    y = data['label']

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(X)

    principalDF = pd.DataFrame(data = principalComponents, columns = ['pc 1', 'pc 2', 'pc 3'])

    finalDf = pd.concat([principalDF, y], axis = 1)

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('pc 1', fontsize = 15)
    ax.set_ylabel('pc 2', fontsize = 15)
    ax.set_zlabel('pc 3', fontsize = 15)

    targets = list(set(y))
    colors = ['#e6194b', '#f58231', '#ffe119', '#bfef45', '#3cb44b', '#42d4f4', '#4363d8', '#911eb4', '#f032e6', '#a9a9a9']
    for target, color in zip(targets,colors):
        keep = finalDf['label'] == target
        ax.scatter(finalDf.loc[keep, 'pc 1']
                , finalDf.loc[keep, 'pc 2']
                , finalDf.loc[keep, 'pc 3']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()
