# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 00:13:53 2021

@author: Sarandeep Singh
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from Functions import accuracy_score, class_wise_accuracy
from matplotlib import pyplot as plt 

train = pd.read_csv("train pixelbypixel.csv")
test = pd.read_csv("test pixelbypixel.csv")
n_classes = 10
train.iloc[:, :-1] = train.iloc[:, :-1]/255
test.iloc[:, :-1] = test.iloc[:, :-1]/255
X, y = train.iloc[:, :-1], train.iloc[:, -1]
X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

model_l1 = LogisticRegression(penalty="l1", solver = "liblinear", C = 1, multi_class="ovr")
model_l1.fit(X, y)
y_pred_train_l1 = model_l1.predict(X)
y_pred_test_l1 = model_l1.predict(X_test)
print("For L1 model")
classwise_train_l1 = class_wise_accuracy(y, y_pred_train_l1, n_classes)
classwise_test_l1 = class_wise_accuracy(y_test, y_pred_test_l1, n_classes)
print("Overall Train accuracy", accuracy_score(y, y_pred_train_l1))
print("Overall Test accuracy", accuracy_score(y_test, y_pred_test_l1))
for i in range(n_classes):
    print("Class", i)
    print("\ttrain", classwise_train_l1[i])
    print("\ttest", classwise_test_l1[i])

model_l2 = LogisticRegression(max_iter = 1000, C = (1/7.5), multi_class="ovr")
model_l2.fit(X, y)
y_pred_train_l2 = model_l2.predict(X)
y_pred_test_l2 = model_l2.predict(X_test)
print("For L2 model")
classwise_train_l2 = class_wise_accuracy(y, y_pred_train_l2, n_classes)
classwise_test_l2 = class_wise_accuracy(y_test, y_pred_test_l2, n_classes)
print("Overall Train accuracy", accuracy_score(y, y_pred_train_l2))
print("Overall Test accuracy", accuracy_score(y_test, y_pred_test_l2))
for i in range(n_classes):
    print("Class", i)
    print("\ttrain", classwise_train_l2[i])
    print("\ttest", classwise_test_l2[i])


#-----------------------C-----------------------
binarized_y_test = label_binarize(y_test, classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
fpr = {}
tpr = {}
roc_auc = {}
y_prob = model_l2.predict_proba(X_test)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(binarized_y_test[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], 'k--')
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label = "class " + str(i))
plt.legend()
plt.show()
