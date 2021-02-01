# -*- coding: utf-8 -*-
"""model_NN_Test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X4GruLAr4KQUCJWnErcqWdIVDYQH6sq5
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2 , f_classif,f_regression,mutual_info_classif,mutual_info_regression
from sklearn.svm import SVR

df = pd.read_csv("/content/drive/MyDrive/ALLAB/smoothed_breast_x_300.csv", delimiter = "\t")
y = pd.read_csv("/content/drive/MyDrive/ALLAB/breast_y.txt.bz2",header=None, delimiter = "\t")

#Zero removed 
# df = df.loc[:, (df != 0).any(axis=0)]
df = df.T

df

k=300

#Anova test
selector = SelectKBest(f_classif, k=k)
selector.fit(df, y)
cols_anova = selector.get_support(indices=True)



cols_to_del = []
for i in range(57914):
    if(i not in cols_anova):
        cols_to_del.append(i)

x = df.drop(columns=cols_to_del)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# define the keras model
model = Sequential()
model.add(Dense(64, input_dim=300, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=15, batch_size=100)

y_pred = model.predict_classes(X_test)

y_pred_seris = pd.Series(y_pred.flatten())


print("---------(Neural Network)-----------")
print(confusion_matrix(y_test,y_pred_seris))
print(classification_report(y_test,y_pred_seris))
print("MCC Score (Neural Network): ",matthews_corrcoef(y_test, y_pred_seris))
print("---------(Neural Network)-----------")