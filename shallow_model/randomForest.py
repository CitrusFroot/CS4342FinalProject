import numpy as np
from sklearn.ensemble import _forest as forest

def rfModel(x, y, split, depth, x2, y2):
    split = np.int64(split)
    xTr = x[0:split, :]
    xTe = x[split: , :]
    yTr = y[0:split]
    yTe = y[split:]
    rf = forest.RandomForestClassifier(max_depth= depth)
    rf.fit(xTr, yTr)
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

def fMSE(y, yhat):
    return np.mean(np.square(np.subtract(yhat, y)))