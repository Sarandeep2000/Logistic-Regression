# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:17:53 2021

@author: Sarandeep Singh
"""
import pandas as pd
import numpy as np

def encode(data, columns):
    for i in columns:
        dummy = pd.get_dummies(data[i], prefix = i, drop_first=True)
        for j in dummy.columns:
            data[j] = dummy[j]
    data.drop(columns = columns, inplace = True)
    return data

def accuracy_score(y_true, y_pred):
    if(type(y_true)!="numpy.ndarray"):
        y_true = y_true.to_numpy()
    arr = np.subtract(y_true, y_pred)
    values, counts = np.unique(arr, return_counts=True)
    index = 0
    inacurate_count = 0
    for i in values:
        if(i != 0):
            inacurate_count+=counts[index]
        index+=1
    return (len(y_true) - inacurate_count)/len(y_true)

def train_test_split(X, y, train_size):
    comb = X
    comb["Income (>50k)"] = y
    column = comb.columns
    comb = comb.to_numpy()
    np.random.shuffle(comb)
    comb = pd.DataFrame(comb, columns = column)
    index = int(train_size * len(comb))
    return comb.iloc[:index, :-1], comb.iloc[index:, :-1], comb.iloc[:index, -1], comb.iloc[index:, -1]

def class_wise_accuracy(y_true, y_pred, n_classes):
    y_true = y_true.to_numpy()
    y = np.zeros(n_classes)
    y_tot = np.zeros(n_classes)
    for i in range(len(y_true)):
            if(y_true[i] == 0):
                y_tot[y_true[i]]+=1
                if(y_pred[i] != y_true[i]):
                    y[y_true[i]]+=1
            elif(y_true[i] == 1):
                y_tot[y_true[i]]+=1
                if(y_pred[i] != y_true[i]):
                    y[y_true[i]]+=1
            elif(y_true[i] == 2):
                y_tot[y_true[i]]+=1
                if(y_pred[i] != y_true[i]):
                    y[y_true[i]]+=1
            elif(y_true[i] == 3):
                y_tot[y_true[i]]+=1
                if(y_pred[i] != y_true[i]):
                    y[y_true[i]]+=1
            elif(y_true[i] == 4):
                y_tot[y_true[i]]+=1
                if(y_pred[i] != y_true[i]):
                    y[y_true[i]]+=1
            elif(y_true[i] == 5):
                y_tot[y_true[i]]+=1
                if(y_pred[i] != y_true[i]):
                    y[y_true[i]]+=1
            elif(y_true[i] == 6):
                y_tot[y_true[i]]+=1
                if(y_pred[i] != y_true[i]):
                    y[y_true[i]]+=1
            elif(y_true[i] == 7):
                y_tot[y_true[i]]+=1
                if(y_pred[i] != y_true[i]):
                    y[y_true[i]]+=1
            elif(y_true[i] == 8):
                y_tot[y_true[i]]+=1
                if(y_pred[i] != y_true[i]):
                    y[y_true[i]]+=1
            elif(y_true[i] == 9):
                y_tot[y_true[i]]+=1
                if(y_pred[i] != y_true[i]):
                    y[y_true[i]]+=1
    accuracy = []
    for i in range(n_classes):
        accuracy.append((y_tot[i] - y_pred[i])/y_tot[i])
    return accuracy
