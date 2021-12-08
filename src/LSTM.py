import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout, Bidirectional, Input, BatchNormalization
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

def LSTM_DL (X_train,y_train ,X_test,y_test):
    """
    WE have utilized yhe LSTM deep leaning model with the following
    Embedding 1000 , spatial_dropout_1d 1000 ,dense 1000 × 5000 , dropout 1000 × 5000  ,lstm 196 ,
    dense 1 600 , dropout 1 600 ,dense 2 100 ,dropout 2 100 ,dense 3 10.WE used
    relu as activation function ,in last Dense layer we took sigmoid since we are dealing with multi-label problem.
    Validation set are used to  process where a trained model is evaluated with a testing data set.
    binary_accuracy used since we convert our multi-label problem to binary.
    :param X_train:Train_Features
    :param y_train:Train_labels
    :param X_test:Test_Features
    :param y_test:Test_labels
    :return:loss, accuracy, f1_score
    """
    lstm_out = 196
    model = Sequential()
    model.add(Embedding(1000, 32, input_length = X_train.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(LSTM(lstm_out, dropout=0.5, recurrent_dropout=0.2))
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy',f1_m])
    model.fit(X_train, y_train,epochs=20,verbose=True,validation_data=0.30,batch_size=10)
    print(model.summary())

    loss, accuracy, f1_score = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    print("f1 :  {:.4f}".format(f1_score))

if __name__ == "__main__":
    # Load Data
    X_train = pd.read_csv(r'../Data/Train_Features.csv', index_col=0)
    y_train = pd.read_csv(r'../Data/Train_Labels.csv', index_col=0)
    X_test = pd.read_csv(r'../Data/Test_Features.csv', index_col=0)
    y_test = pd.read_csv(r'../Data/Test_Labels.csv', index_col=0)
    #pass the data to the model after retrived them from excel sheet
    LSTM_DL(X_train,y_train,X_test,y_test)
