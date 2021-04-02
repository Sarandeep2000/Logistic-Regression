# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 20:19:17 2021

@author: Phoenix
"""

import pickle
import pandas as pd
import numpy as np

def generate_frame(dataset, columns):
    pic = dataset.iloc[0, 0]
    arr = np.array(pic.getdata()).reshape(1, pic.size[0]*pic.size[1])
    for i in range(1, len(dataset)):
        arr = np.append(arr, np.array(dataset.iloc[i, 0].getdata()).reshape(1, pic.size[0]*pic.size[1]), axis = 0)
    del pic
    return pd.DataFrame(arr, columns=columns)

f_train = open("train_set.pkl", "rb")
f_test = open("test_set.pkl", "rb")
train_data = pickle.load(f_train)
test_data = pickle.load(f_test)
f_train.close()
f_test.close()
pic = train_data.iloc[0, 0]

columns = []
for i in range(pic.size[0]):
    for j in range(pic.size[1]):
        columns.append("pixel"+ str(i) + "by" +str(j))
del pic

train_frame = generate_frame(train_data, columns)
train_frame["Labels"] = train_data["Labels"]
test_frame = generate_frame(test_data, columns)
test_frame["Labels"] = test_data["Labels"]

train_frame.to_csv("train pixelbypixel.csv")
test_frame.to_csv("test pixelbypixel.csv")