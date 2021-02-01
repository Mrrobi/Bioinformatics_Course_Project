# -*- coding: utf-8 -*-
"""model_NN_test(cancer+normal).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IFn5k6ljM3tfWNZ_eJPaKh8FAY0i2LcO
"""

# first neural network with keras tutorial
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
Cancer = pd.read_csv("/content/drive/MyDrive/ALLAB/cancer_bin.txt.bz2",header=None, delimiter = "\t")
Normal = pd.read_csv("/content/drive/MyDrive/ALLAB/normal_bin.txt.bz2",header=None, delimiter = "\t")
Cancer['Target'] = 1
Normal['Target'] = 0
Normal = Normal.drop(Normal.index[0])
frame = [Cancer,Normal]
Data = pd.concat(frame,axis=0)
Data = Data.drop(Data.index[0])
x = Data.iloc[:,:300]
y = Data.iloc[:,300]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# define the keras model
model = Sequential()
model.add(Dense(64, input_dim=300, activation='relu'))
model.add(Dense(32, activation='relu')) #hidden layer #1
model.add(Dense(16, activation='relu')) #hidden layer #2
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