import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def load_data(strin, scaler, orient):
    #loading csv into a pandas dataframe
    grc = pd.read_csv(strin)
    x = grc[["Erm/Sci","Scm/Po","conf"]]
    print("X features: ")
    print(x.head())
    y = grc[["strain"]]
    print("Y features: ")
    print(y.head())

    #converting pd dataframe into a numpy array for neural network
    X = x.to_numpy()
    Y = y.to_numpy()

    #spllitting the array into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #pre processing the data
    if(scaler=="MinMax"):
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()


    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    if(orient):
        X_train = X_train.T
        print("shape of X_train: ",X_train.shape)
        y_train = y_train.reshape(1,X_train.shape[1])
        print("shape of y_train: ",y_train.shape)

        X_test = X_test.T
        print("shape of X_test: ",X_test.shape)
        y_test = y_test.reshape(1,-1)
        print("shape of y_test: ",y_test.shape) 

    return X_train, X_test, y_train, y_test

def load_dataLDP(strin, scaler, orient):
    #loading csv into a pandas dataframe
    ldp = pd.read_csv(strin)
    x = ldp[["Erm/Sci","Scm/Po","X*"]]
    print("X features: ")
    print(x.head())
    y = ldp[["strain"]]
    print("Y features: ")
    print(y.head())

    #converting pd dataframe into a numpy array for neural network
    X = x.to_numpy()
    Y = y.to_numpy()

    #spllitting the array into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #pre processing the data
    if(scaler=="MinMax"):
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()


    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    if(orient):
        X_train = X_train.T
        print("shape of X_train: ",X_train.shape)
        y_train = y_train.reshape(1,X_train.shape[1])
        print("shape of y_train: ",y_train.shape)

        X_test = X_test.T
        print("shape of X_test: ",X_test.shape)
        y_test = y_test.reshape(1,-1)
        print("shape of y_test: ",y_test.shape) 

    return X_train, X_test, y_train, y_test