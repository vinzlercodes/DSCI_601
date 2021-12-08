import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D,MaxPooling1D,BatchNormalization
from keras import backend as K

def recall_m(y_true, y_pred):
    """
    Metrics have been removed from Keras core. we need to calculate them manually
    here we will calculate the recall score
    :param y_true:is the true data (or target, ground truth)
    :param y_pred:is the data predicted (calculated, output) by your model.
    :return: recall score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    """
    Metrics have been removed from Keras core. we need to calculate them manually
    here we will calculate the precision score
    :param y_true:is the true data (or target, ground truth)
    :param y_pred:is the data predicted (calculated, output) by your model.
    :return: precision score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    """
    Metrics have been removed from Keras core. we need to calculate them manually
    here we will calculate the f1 macro score
    :param y_true:is the true data (or target, ground truth)
    :param y_pred:is the data predicted (calculated, output) by your model.
    :return: f1 macro score
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def CNN (X_train,y_train ,X_test,y_test):

    """
    Here we have utilized the Convolutional Neural Network deep learning model , we ave used 6 layers with
    relu as activation function and last Dense layer we took sigmoid since we are dealing with multi-label problem.
    Validation set are used to  process where a trained model is evaluated with a testing data set.
    binary_accuracy used since we convert our multi-label problem to binary.
    :param X_train:Train_Features
    :param y_train:Train_labels
    :param X_test:Test_Features
    :param y_test:Test_labels
    :return:loss, accuracy, f1_score
    """
    input_dim = X_train.shape[1]  # Number of features
    model = Sequential()
    model.add(Dense(5000, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(60, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(40, activation="relu"))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy',f1_m])
    model.summary()
    model.fit(X_train, y_train,epochs=20,verbose=True,validation_data=(X_test, y_test),batch_size=10)

    loss, accuracy, f1_score = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    print("f1 :  {:.4f}".format(f1_score))


if __name__ == "__main__":
     # Load data
    X_train = pd.read_csv(r'../Data/Train_Features.csv', index_col=0)
    y_train = pd.read_csv(r'../Data/Train_Labels.csv', index_col=0)
    X_test = pd.read_csv(r'../Data/Test_Features.csv', index_col=0)
    y_test = pd.read_csv(r'../Data/Test_Labels.csv', index_col=0)
    #pass the data to the model after retrived them from excel sheet
    CNN(X_train,y_train,X_test,y_test)
