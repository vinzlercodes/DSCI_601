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
            stop_words = stopwords.words('english')
            lemmatizer = WordNetLemmatizer()
            x = x.lower()
            x = x.translate(str.maketrans('', '', string.punctuation))
            x = x.split()
            x = [word for word in x if word not in stop_words]
            x = [lemmatizer.lemmatize(word) for word in x]
            x = str(x).replace(',', ' ').replace("'", "")[1:-1]
            return x
        except:
            print(f'There is an error in {x}')
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

    def Vectorization(self, df):
        """
        This function applied Vectorization using TF-IDF  here for each word cell to achieve
        a weight importance value to a particular word in the list and
        that will help with highlighting certain syntax words or indicative words
        that will help with refactoring label prediction
        :param df:this is the pandas data frame use it with text column to apply Tfidf Vectorizer
        :return:thus will return x as Vectorized text
        """
        v = TfidfVectorizer(max_features=1000)
        x = v.fit_transform(df['message'])

        return x, v

    def concatnate(self, df):
        """
        This function cleans the 'message' column and then replaces the values with the vectorized text for better
        class to input correlation

        :param x: vectorized dataframe 'text' column
        :param df: the cleaned and dataframe in use
        :return df: preprocessed dataframe with vectorized 'message' column
        """

        df.dropna(subset=["message"], inplace=True)
        df = df.drop(columns=['description','detection_tool','type'])
        df['message'] = df['message'].apply(self.preprocess)

        return df

    def __call__(self):
        """
        The __call__ method used here to turn the instances
        of the class into callables. where here the instances behave like
        functions and can be called like a function to be implemented.
        :return:
        """
        # this line will drop the null values of text column
        self.dataFrame.dropna(subset=["message"], inplace=True)
        # here to drop duplicate values of text column
        self.dataFrame = self.dataFrame.drop_duplicates(subset=['message'])
        # implement labelBinarizer function
        self.classes = self.labelBinarizer(self.dataFrame['type'])

        self.dataFrame = self.dataFrame.reset_index(drop=True)
        self.dataFrame = pd.concat([self.dataFrame, self.classes], axis=1)
        # apply Vectorization tfidf function and concatenate the data
        #vectorOfFeatures, self.vectorzier = self.Vectorization(self.dataFrame)
        self.dataFrame = self.concatnate( self.dataFrame)
        # saved the Preprocessed data
        self.dataFrame.to_csv("../MSR 2022/Preprocessed.csv")

if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv(r'../MSR 2022/result_mongo.csv')
    dataprep = DataPreparation(data)
    dataprep()
    print('The data is ready')
