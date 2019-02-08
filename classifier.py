from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import os

MODEL_FILE_NAME = 'nbclassifier.sav'
BASE_PATH = os.path.dirname(os.path.realpath(__file__))

DATA_FILE = BASE_PATH + os.sep + 'data.csv'


def train_nb(x_train, y_train):
    '''Trains the naive bayes classifier with dataset
    x_train is the list of string y_train ss the set of
    labels'''
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    pickle.dump(gnb, open(MODEL_FILE_NAME, 'wb'))  # Since gnb is our model


def predict(y_test=[2, 2, 0, 0]):
    '''Takes a list as input'''
    loaded_gnb_model = pickle.load(open(MODEL_FILE_NAME, 'rb'))
    y_test = np.array([y_test])
    predicted_value = loaded_gnb_model.predict(y_test)
    print('Predicted value', predicted_value)


def load_data(data_file='data.csv'):
    df = pd.read_csv(data_file)
    categorical_data = df.apply(LabelEncoder().fit_transform)
    x_train = categorical_data.iloc[:, :-1]
    y_train = categorical_data.iloc[:, -1]
    y_train = np.array(y_train)
    x_train = np.array(x_train)
    y_train = y_train.reshape(-1, 1)
    return x_train, y_train


x_train, y_train = load_data(DATA_FILE)


train_nb(x_train, y_train)
predict(y_test=[2, 2, 0, 0])
