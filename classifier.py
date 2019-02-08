from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def train_nb(x_train, y_train):
    '''Trains the naive bayes classifier with dataset
    x_train is the list of string y_train ss the set of
    labels'''
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_test = np.array([[2, 2, 0, 0]])

    predicted_value = gnb.predict(y_test)
    print('Predicted value', predicted_value)


def load_data(data_file='data.csv'):
    df = pd.read_csv(data_file)
    categorical_data = df.apply(LabelEncoder().fit_transform)
    x_train = categorical_data.iloc[:, :-1]
    y_train = categorical_data.iloc[:, -1]
    y_train = np.array(y_train)
    x_train = np.array(x_train)
    y_train = y_train.reshape(-1, 1)
    # ohe = OneHotEncoder(sparse=False)
    # y_train = ohe.fit_transform(y_train)
    print('Shape of x', x_train.shape)
    print('Shape of y', y_train.shape)

    return x_train, y_train

    '''
    with open(data_file, 'r') as file_handle:
        for lines in file_handle.readlines():
            lines = lines.strip('\n').split(',')
            # print(y_train, lines[-1])
            np.append(y_train, lines[-1])
            np.append(x_train, lines[:-1])
    '''

    return x_train, y_train


x_train, y_train = load_data()


# print('x_train -> ', x_train)
# print('y_train -> ', y_train)


for x, y in zip(x_train, y_train):
    print(x, y)
train_nb(x_train, y_train)
