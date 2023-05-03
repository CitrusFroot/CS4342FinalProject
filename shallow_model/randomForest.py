import numpy as np
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.tree import plot_tree
import csv
import matplotlib.pyplot as plt
import joblib

def rfModel(x, y, split, depth, estimators, x2, y2):
    split = np.int64(split)
    xTr = x[0:split, :] #training set
    xTe = x[split: , :] #testing set
    yTr = y[0:split]
    yTe = y[split:]
    rf = RandomForestClassifier(max_depth= depth, n_estimators = estimators)
    rf.fit(xTr, yTr) #creates model
    yhatTr = rf.predict(xTr)
    yhatTe = rf.predict(xTe)
    yhat2 = rf.predict(x2)
    print('=======RANDOM FOREST RESULTS:=========')
    print('training accuracy:', rf.score(xTr, yTr))
    print('training MSE     :', fMSE(yTr, yhatTr))
    print('testing accuracy :', rf.score(xTe, yTe))
    print('testing MSE      :', fMSE(yTe, yhatTe))
    print('\n  === Dig-MNIST Predictions ===')
    print('accuracy :', rf.score(x2, y2))
    print('MSE      :', fMSE(y2, yhat2))
    print('======================================')

    joblib.dump(rf, "randomForestModel.joblib") #saves model
    generateCSV(rf)
    

def fMSE(y, yhat):
    return np.mean(np.square(np.subtract(yhat, y)))

def generateCSV(rf):
    test = np.loadtxt('Kannada-MNIST/test.csv', delimiter=',', skiprows = 1) #skip header row
    id = test[:, 0] #first column = id
    x = test[:, 1:] #shape = (#imgs, pixels)
    yhat = rf.predict(x)
    id = np.int64(id)
    yhat = np.int64(yhat)
    submission = np.transpose([id,yhat])

    file = open('randomForestSubmission.csv', 'w', newline= '')
    writer = csv.writer(file)
    writer.writerow(['id','label'])
    writer.writerows(submission)
    file.close()
    
def visualizeRF():
    rf = joblib.load("randomForestModel.joblib")
    dTree = rf.estimators_[0] #gets first decision tree
    fig = plt.figure(figsize=(15,10))
    _ = plot_tree(dTree, filled=True, rounded = True, impurity = True)
    fig.savefig('rfTree.png')