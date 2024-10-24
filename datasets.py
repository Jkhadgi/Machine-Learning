# -*- coding: utf-8 -*-
"""
Created on Mon May  1 22:10:44 2023

@author: admin
"""

#importing the libraries
import os
#import pandas as pd


import os

def datasets_folder(dir):
    data = []  
    mytest_data = []

    for folder in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, folder)):
            if folder == 'mytest':
                for file in os.listdir(os.path.join(dir, folder)):
                    if file.endswith(".txt"):
                        file_path = os.path.join(dir, folder, file)   
                        with open(file_path, "r") as f:
                            text = f.read()
                            mytest_data.append((text))
            else:
                for file in os.listdir(os.path.join(dir, folder)):
                    if file.endswith(".txt"):
                        file_path = os.path.join(dir, folder, file)   
                        with open(file_path, "r") as f:
                            text = f.read()
                            data.append((text,folder))
    return data, mytest_data






###########################################################################
from sklearn import datasets

def iris_dataset():
    iris = datasets.load_iris()
    x = iris.data    #petal width, petal length, sepal width, septal length...
    y = iris.target   #species name 
    return x, y




'''
from sklearn.model_selection import train_test_split
def iris_dataset():
    # Load the iris dataset
    iris = datasets.load_iris()

    #For the binary classification, use first two features and only [0,1] labels in the dataset
    X = iris.data[iris.target < 2, :2]
    y = iris.target[iris.target < 2]

    # Splitting the dataset
    tr_x, val_x, tr_y, val_y = train_test_split(X, y, test_size=0.2)

    return tr_x, tr_y, val_x, val_y
'''

