from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras import callbacks, models
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle


ADDRESS = 'Put the address to all the images here'
BATCHSIZE = 32
EPOCHCOUNT = 1
IMAGESIZE = 28
VALIDATIONSPLIT = 0.1   
CLASSCOUNT = 10
INPUT_SHAPE = [IMAGESIZE, IMAGESIZE, 1]

def getDigSet():
    dataset = np.loadtxt('CS4342FinalProject/Kannada-MNIST/Dig-MNIST.csv', delimiter=',', skiprows = 1)
    labels = dataset[:, 0] #first column = y
    dataset = dataset[:, 1:] #shape = (#imgs, pixels)
    
    # normalize image pixel
    dataset = dataset.reshape(-1, IMAGESIZE, IMAGESIZE, 1) / 255.0     

    # load pre-categorized labels
    # labels = np.loadtxt('val_y_labels.csv')
    labels = oneHotEncoding(labels)

    return dataset, labels

def getTrainSet():
    dataset = np.loadtxt('CS4342FinalProject/Kannada-MNIST/train.csv', delimiter=',', skiprows = 1)
    labels = dataset[:, 0] #first column = y
    dataset = dataset[:, 1:] #shape = (#imgs, pixels)
    
    # normalize image pixel
    dataset = dataset.reshape(-1, IMAGESIZE, IMAGESIZE, 1) / 255.0    

    # load pre-categorized labels
    # labels = np.loadtxt('y_labels.csv')
    labels = oneHotEncoding(labels)
    
    return dataset, labels

def getTestSet():
    dataset = np.loadtxt('CS4342FinalProject/Kannada-MNIST/test.csv', delimiter=',', skiprows = 1)
    id = dataset[:, 0] #first column = y
    dataset = dataset[:, 1:] #shape = (#imgs, pixels)

    # normalize image pixel
    dataset = dataset.reshape(-1, IMAGESIZE, IMAGESIZE, 1) / 255.0    

    return dataset, id

# Convert Categorical Value to One-Hot encoding 
def oneHotEncoding(y):
    m = np.shape(y)[0]
    print('m = ', m)
    output = np.zeros((m, 10))
    for r in range(0, m):
        index = (int)(y[r])
        output[r, index] = 1
    return output

# Convert One-Hot encoding to Categorical Value
def convertGroundTruth(yhat):
    m = np.shape(yhat)[0]
    output = np.zeros((m, 1))
    for r in range(0, m):
        maxind = np.argmax(yhat[r,:])
        output[r] = maxind
    return output


def three_layer_NN():

    # --- Get all datasets -- #
    x, y = getTrainSet()
    test_x, test_id = getTestSet()
    dig_x, dig_y = getDigSet()

    x_train, x_val, y_train, y_val = train_test_split(x, y)

    # --- Build Sequential Model --- #

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=5, padding='same',input_shape=INPUT_SHAPE, activation='relu'))
    model.add(Flatten())
    model.add(Dense(CLASSCOUNT, activation="softmax"))

    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # model.save("NN model")
    # with open('NNhistory', "rb") as file_pi:
        # hist = pickle.load(file_pi)
    
    earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=EPOCHCOUNT, batch_size=BATCHSIZE, validation_data=(x_val, y_val), callbacks=[earlystopping], shuffle=True)


    model.summary()

    # --- Accuracy --- # 

    # training set
    yhat = model.predict(x_train)
    r2_score_train = r2_score(y_train, yhat)
    mse_score_train = mean_squared_error(y_train, yhat)

    print('r2_score_train = ', r2_score_train)
    print('mse_score_train = ', mse_score_train)

    # validation set
    yhat_val = model.predict(x_val)
    r2_score_val = r2_score(y_val, yhat_val)
    mse_score_val = mean_squared_error(y_val, yhat_val)

    print('r2_score_val = ', r2_score_val)
    print('mse_score_val = ', mse_score_val)

    # dig set
    yhat_dig = model.predict(dig_x)
    r2_score_dig = r2_score(dig_y, yhat_dig)
    mse_score_dig = mean_squared_error(dig_y, yhat_dig)

    print('r2_score_dig = ', r2_score_dig)
    print('mse_score_dig = ', mse_score_dig)

    # --- Plotting --- #
    err = hist.history['accuracy']
    val_err = hist.history['val_accuracy']

    epochs = range(1, len(err) + 1)

    plt.plot(epochs, err, '-', label='Training Accuracy')
    plt.plot(epochs, val_err, ':', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend(loc='upper right')
    plt.plot()
    plt.show()

    test_y = model.predict(test_x)
    prediction = convertGroundTruth(test_y)

    # --- Converting to CSV for Kaggle Submission --- #

    # id_col = np.arange(prediction.shape[0])
    # submission = pd.DataFrame({'id': id_col, 'label': prediction})
    # submission.to_csv('sub.csv', index = False)
    
    # --- Notes --- #
    
    # epoch = 20; neuron = 500; batchSize = 150; loss accuracy = 0.9X; validation accuray = 0.15
    # epoch = 20; neuron = 200; batchSize = 150; loss accuracy = 0.92; validation accuray = 0.27
    # epoch = 20; neuron = 150; batchSize = 150; loss accuracy = 0.85; validation accuray = -0.22
    # epoch = 20; neuron = 100; batchSize = 150; loss accuracy = 0.93; validation accuray = 0.17

    # neuron = 200

    # epoch = 20; neuron = 200; batchSize = 320; loss accuracy = 0.82; validation accuray = -0.10
    # epoch = 20; neuron = 200; batchSize = 600; loss accuracy = 0.36; validation accuray = -3.24

    # lower batchsize

    # epoch = 20; neuron = 200; batchSize = 30; loss accuracy = 0.94; validation accuray = 0.35 
    # epoch = 20; neuron = 200; batchSize = 15; loss accuracy = 0.92; validation accuray = 0.33
    # epoch = 20; neuron = 200; batchSize = 60; loss accuracy = 0.96; validation accuray = 0.44

    # sigmoid
    # epoch = 20; neuron = 200; batchSize = 60; loss accuracy = 0.96; validation accuray = 0.44

    # epoch = 20; neuron = 200; batchSize = 100; loss accuracy = 0.97; validation accuray = 0.48
    # epoch = 20; neuron = 200; batchSize = 120; loss accuracy = 0.97; validation accuray = 0.50
    
    # epoch = 50; neuron = 500; batchSize = 120; loss accuracy = 0.98; validation accuray = 0.56

    # conv2d w/ reLu (final)
    # epoch = 20; conv2d = 64; batchsSize = 32; training accuracy = 0.9973 ; validation accuracy = 0.9988


    