__authors__ = 'Abdullah + Vinayak'

import pickle
import numpy as np
import pandas as pd

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