import pickle
import numpy as np
import pandas as pd
import time
from sklearn.metrics import classification_report

def train_and_pickle(model,name,X_train,y_train):
    """
    :param model:
    :param name:
    :param X_train:
    :param y_train:
    :return:
    """
    model.fit(X_train,y_train)
    print(X_train)
    y_pred = model.predict(X_train)

    print(classification_report(y_pred,y_train))
    with open('../models/'+name+'.pickle', 'wb') as f:
        pickle.dump(model, f)