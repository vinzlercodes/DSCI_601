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

        :param df:
        :return:
        """
        mlb = MultiLabelBinarizer()
        tempdf = pd.DataFrame(columns=['labels'])
        for i in df:
            temp = []
            try:
                i = i.replace(' ', '')
                for j in i.split(','):
                    if j != '':
                        temp.append(j.strip())
            except:
                pass
            tempdf = tempdf.append(pd.DataFrame({'labels': [temp]}))

        tempdf.apply(lambda x: tuple(x.values))
        mlb.fit(tempdf['labels'])
        tempdf = mlb.transform(tempdf['labels'])
        tempdf = pd.DataFrame(tempdf, columns=list(mlb.classes_))
        return tempdf

    def Vectorization(self, df):
        """
        This function applied Vectorization using TF-IDF  here for each word cell to achieve
        a weight importance value to a particular word in the list and
        that will help with highlighting certain syntax words or indicative words
        that will help with refactoring label prediction
        :param df:this is the pandas data frame use it with text column to apply TfidfVectorizer
        :return:thus will return x as Vectorized text
        """
        v = TfidfVectorizer(max_features=1000)
        x = v.fit_transform(df['Text'])

        return x,v

    def concatnate(self, x, df):
        """

        :param x:
        :param df:
        :return:
        """
        df.dropna(subset=["Text"], inplace=True)
        df = df.drop(columns=['ID', 'Class'])
        df['Text'] = df['Text'].apply(self.preprocess)
        df = pd.concat([pd.DataFrame(x.toarray()), df], axis=1)

        return df

    def split(self, df):
        """

        :param df:
        :return:
        """
        columns = list(df.columns)
        X_train, X_test, y_train, y_test = train_test_split(df[columns[:-4]], df[columns[-4:]], test_size=0.25,
                                                            random_state=42)
        X_train = X_train.drop(columns=['Text'])
        X_test = X_test.drop(columns=['Text'])
        X_train.to_csv(r'../Data/Train_Features.csv')
        y_train.to_csv(r'../Data/Train_Labels.csv')
        X_test.to_csv(r'../Data/Test_Features.csv')
        y_test.to_csv(r'../Data/Test_Labels.csv')

        return X_train, X_test, y_train, y_test

    def __call__(self):
        """
        The __call__ method used here to turn the instances
        of the class into callables. where here the instances behave like
        functions and can be called like a function to be implemented.
        :return:
        """
        #this line will drop the null values of text column
        self.dataFrame.dropna(subset=["Text"], inplace=True)
        #here to drop dublicate values of text column
        self.dataFrame = self.dataFrame.drop_duplicates(subset=['Text'])
        #implement labelBinarizer function
        self.classes = self.labelBinarizer(self.dataFrame['Class'])
        #this will rest the index before concatnate the data
        self.dataFrame = self.dataFrame.reset_index(drop=True)
        self.dataFrame = pd.concat([self.dataFrame, self.classes], axis=1)
        #saved the Preprocessed data
        self.dataFrame.to_csv("../Data/Preprocessed.csv")
        #apply Vectorization tfidf functiom and concatnate the data
        vectorOfFeatures,self.vectorzier = self.Vectorization(self.dataFrame)
        self.dataFrame = self.concatnate(vectorOfFeatures,self.dataFrame)
        #split the data set
        self.X_train,self.X_test,self.y_train,self.y_test = self.split(self.dataFrame)

if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv(r'../Data/FR-Dataset.csv')
    dataprep = DataPreparation(data)
    dataprep()
    print('The data is ready')


