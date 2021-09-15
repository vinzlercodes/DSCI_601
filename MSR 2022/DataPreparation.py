__authors__ = 'Abdullah + Vinayak'

import pandas as pd
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class DataPreparation:

    def __init__(self, dataFrame):
        """
        The method initialises the variables and the data frame and the other parameters that will be utilised.
        :param dataFrame: the raw dataset 'FR-dataset'
        """
        self.dataFrame = dataFrame
        self.classes = None
        self.vectorzier = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def preprocess(self, x):
        """
        This function responsible of preprocess for the Text column starting by removing the stop word
        and using lemmatization technique from nltk library , also remove punctuation and the comma
        and quotation marks
        :param x: x here is  the text column in the dataset , this param uses for applying the preprocess steps
        :return: this will return the column after been preprocessed  or return empty if there is an error
        """
        try:
            lemmatizer = WordNetLemmatizer()
            stop_words = stopwords.words('english')
            x = x.lower()
            x = x.translate(str.maketrans('', '', string.punctuation))
            x = x.split()
            x = [word for word in x if word not in stop_words]
            x = [lemmatizer.lemmatize(word) for word in x]
            x = str(x).replace(',', ' ').replace("'", "")[1:-1]
            return x
        except:
            print(f'There is an error while preprocess in {x}')
            return 'empty'

    def labelBinarizer(self, df):
        """
        The function converts the multilabel problem into binary classification across multiple classes.
        The dataset classes are converted into unique columns and their presence values are encoded by 0 or 1.
        :param df: the cleaned and lemmatized dataframe
        :return tempdf: a further preprocessed dataframe with dummy variables
        """
        mlb = MultiLabelBinarizer()
        # select the 'labels' column for dummy creation
        tempdf = pd.DataFrame(columns=['labels'])
        for i in df:
            temp = []
            try:
                # separating the classes by whitespace
                i = i.replace(' ', '')
                # separating each class entry using ',' delimeter
                for j in i.split(','):
                    if j != '':
                        temp.append(j.strip())
            except:
                pass
            tempdf = tempdf.append(pd.DataFrame({'labels': [temp]}))
        # storing the classes for each entry in tuples
        tempdf.apply(lambda x: tuple(x.values))
        mlb.fit(tempdf['labels'])
        # creating the dummy variables for each unique class
        tempdf = mlb.transform(tempdf['labels'])
        tempdf = pd.DataFrame(tempdf, columns=list(mlb.classes_))
        return tempdf

    def __call__(self):
        """
        The __call__ method used here to turn the instances
        of the class into callables. where here the instances behave like
        functions and can be called like a function to be implemented.
        :return:
        """
        # this line will drop the null values of text column
        self.dataFrame.dropna(subset=["Text"], inplace=True)
        # here to drop duplicate values of text column
        self.dataFrame = self.dataFrame.drop_duplicates(subset=['Text'])
        # implement labelBinarizer function
        self.classes = self.labelBinarizer(self.dataFrame['Class'])

if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv(r'C:\Users\vinay\Desktop\DSCII 601\Project\DSCI_601\MSR_2022\result_mongo.csv')
    dataprep = DataPreparation(data)
    dataprep()
    print('The data is ready')
