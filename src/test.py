__authors__ = 'Abdullah + Vinayak'

import pickle
import pandas as pd
import time
from sklearn.metrics import classification_report

def unpickle_and_test(name,X_test,y_test):
    """
    :param name:
    :param X_test:
    :param y_test:
    :return:
    """
    loaded_obj = None
    with open('../models/'+name+'.pickle','rb') as f:
        loaded_obj = pickle.load(f)
    assert loaded_obj is not None
    y_pred = loaded_obj.predict(X_test)
    print(classification_report(y_pred,y_test))

if __name__ == '__main__':

    # Load data
    X_test = pd.read_csv(r'../Data/Test_Features.csv',index_col=0)
    y_test = pd.read_csv(r'../Data/Test_labels.csv',index_col=0)
    classifiers = ['RandomForest', 'MNB','SVM','LR']
    for x in classifiers:
        start = time.time()
        print(x)
        unpickle_and_test(x,X_test,y_test)
        print((time.time()-start) , 'sec')
