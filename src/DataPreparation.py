import pandas as pd
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class DataPreparation:

    def __init__(self,dataFrame):
        """

        :param dataFrame:
        """
        self.dataFrame = dataFrame
        self.classes = None
        self.vectorzier = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None