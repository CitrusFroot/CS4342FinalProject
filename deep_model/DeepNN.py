from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization, Flatten
from keras import callbacks, models
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

BATCHSIZE = 32
EPOCHCOUNT = 30
IMAGESIZE = 28
INPUT_SHAPE = [IMAGESIZE, IMAGESIZE, 1]

def deepNN():
    train_df = pd.read_csv('Kannada-MNIST/train.csv')
    dig_df = pd.read_csv('Kannada-MNIST/Dig-MNIST.csv')

    train_img = train_df.drop('label', axis=1).values
    train_img = train_img.reshape(-1, IMAGESIZE, IMAGESIZE, 1) / 255.0

    x_train, x_val, y_train, y_val = train_test_split(train_img, train_df['label'])

    x_dig = dig_df.drop('label', axis=1).values
    x_dig = x_dig.reshape(-1, IMAGESIZE, IMAGESIZE, 1) / 255.0

    y_dig = dig_df['label']

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=5, padding='same',input_shape=INPUT_SHAPE, activation='relu'))
    model.add(MaxPool2D(pool_size=3))
    model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(10, activation="softmax"))

    earlystopping = callbacks.EarlyStopping(monitor="accuracy", mode="max", patience=5, restore_best_weights=True)

    #*********************Train and save the model***************#
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # hist = model.fit(x_train, y_train, epochs=EPOCHCOUNT, batch_size=BATCHSIZE, validation_data=(x_val, y_val), callbacks=[earlystopping])

    # model.save("Deep Learning model")

    # with open('deepNNHistory', 'wb') as file_pi:
    #     pickle.dump(hist.history, file_pi)
    

    #***************Load up saved model and history************#
    model = models.load_model('deep_model/Deep Learning model')
    with open('deep_model/deepNNHistory', "rb") as file_pi:
        hist = pickle.load(file_pi)
    
    # model.summary()

    plot_model(model, to_file="deep_model/Deep_NN_Model.png", show_shapes=True)

    pred_train = np.argmax(model.predict(x_train), axis=1)
    r2_score_train = r2_score(y_train, pred_train)
    mse_score_train = mean_squared_error(y_train, pred_train)

    pred_val = np.argmax(model.predict(x_val), axis=1)
    r2_score_val = r2_score(y_val, pred_val)
    mse_score_val = mean_squared_error(y_val, pred_val)

    pred_dig = np.argmax(model.predict(x_dig), axis=1)
    r2_score_dig = r2_score(y_dig, pred_dig)
    mse_score_dig = mean_squared_error(y_dig, pred_dig)

    print('=======DEEP NN RESULTS:=========')
    print("Training Acc: {}".format(round(r2_score_train, 4)))
    print("Training MSE: {}".format(round(mse_score_train, 4)))
    print("Validation Acc: {}".format(round(r2_score_val, 4)))
    print("Validation MSE: {}".format(round(mse_score_val, 4)))
    print("Dig Acc: {}".format(round(r2_score_dig, 4)))
    print("Dig MSE: {}".format(round(mse_score_dig, 4)))
    print('======================================')

    # acc = hist['accuracy']
    # val_acc = hist['val_accuracy']

    # epochs = range(1, len(acc) + 1)

    # plt.plot(epochs, acc, '-', label='Training Acc')
    # plt.plot(epochs, val_acc, ':', label='Validation Acc')
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend(loc='lower right')
    # plt.plot()
    # plt.show()

    #***************Write Submission file************#
    # test_df = pd.read_csv('Kannada-MNIST/test.csv')
    # test_img = test_df.drop('id', axis=1).values.reshape(-1,28,28,1)/255.0

    # preds = np.argmax(model.predict(test_img), axis=1)
    # id = test_df['id']

    # submission = np.column_stack((np.array(id), preds))
    # submission = submission.astype(int)
    # np.savetxt('deepNNsubmission.csv', submission, delimiter=',', header='id,label', comments='', fmt='%i')

    