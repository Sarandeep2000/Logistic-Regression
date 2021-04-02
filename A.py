# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:14:35 2021

@author: Sarandeep Singh
"""

from LogisticRegression import Logistic_Regression
from Functions import encode, accuracy_score, train_test_split
import pandas as pd
from time import time

start = time()
df1 = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")
df1.insert(loc = 0, column = "intercept", value = 1)
df2.insert(loc = 0, column = "intercept", value = 1)

#preprocessing
df1.drop(columns=["education-num"], inplace = True)
df2.drop(columns=["education-num"], inplace = True)

df1_mean = df1.mean()
df1_std = df1.std()
for i in df1_mean.index:
    df1[i] = (df1[i] - df1_mean[i])/df1_std[i]
    df2[i] = (df2[i] - df1_mean[i])/df1_std[i]

cat_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "Income"]

#The following part of the code ensures that the same features are made
#while one-hot encoding for both train.csv and test.csv
train_len = len(df1)
merge_df = pd.concat(objs = [df1, df2], axis = 0)
merge_df = encode(merge_df, cat_columns)
train_df = merge_df[:train_len]
test_df = merge_df[train_len:]
del merge_df

X, y = train_df.iloc[:, :-1], train_df.iloc[:, -1]
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.8)
X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]
X_train.reset_index(inplace = True, drop = True)
y_train.reset_index(inplace = True, drop = True)
X_val.reset_index(inplace = True, drop = True)
y_val.reset_index(inplace = True, drop = True)

model_l1 = Logistic_Regression(iterations = 1000, r_type = "L1", alpha = 0.25, lamb = 3.5)
model_l2 = Logistic_Regression(iterations = 1000, alpha = 0.25, lamb = 3.5)
model_l1.fit(X_train, y_train)
print("model 1 fitted")
model_l2.fit(X_train, y_train)
print("model 2 fitted")

y_pred_train_l1 = model_l1.predict(X_train)
y_pred_train_l2 = model_l2.predict(X_train)
print("Train set accuracy for L1 regularizer:", accuracy_score(y_train, y_pred_train_l1))
print("Train set accuracy on L2 regularizer:", accuracy_score(y_train, y_pred_train_l2))

y_pred_val_l1 = model_l1.predict(X_val)
y_pred_val_l2 = model_l2.predict(X_val)
print("Validation set accuracy on L1 regularizer:", accuracy_score(y_val, y_pred_val_l1))
print("Validation set accuracy on L2 regularizer:", accuracy_score(y_val, y_pred_val_l2))

y_pred_test_l1 = model_l1.predict(X_test)
y_pred_test_l2 = model_l2.predict(X_test)
print("Test set accuracy on L1 regularizer:", accuracy_score(y_test, y_pred_test_l1))
print("Test set accuracy on L2 regularizer:", accuracy_score(y_test, y_pred_test_l2))

model_l1.plot_log_loss_vs_iteration()
model_l1.plot_loss_vs_iteration()
model_l1.plot_accuracy_vs_iteration()
model_l2.plot_log_loss_vs_iteration()
model_l2.plot_loss_vs_iteration()
model_l2.plot_accuracy_vs_iteration()
