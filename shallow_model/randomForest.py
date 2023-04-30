import numpy as np
from sklearn.ensemble import _forest as forest

def rfModel(x, y, split):
    split = np.int64(split)
    xTr = x[0:split, :]
    xTe = x[split: , :]
    yTr = y[0:split]
    yTe = y[split:]
    rf = forest.RandomForestClassifier(max_depth= 45)
    rf.fit(xTr, yTr)
    print('=======RANDOM FOREST RESULTS:=========')
    print('training accuracy:', rf.score(xTr, yTr))
    print('testing accuracy :', rf.score(xTe, yTe))
    print('======================================')

