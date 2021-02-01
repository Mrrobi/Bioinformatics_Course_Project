# -*- coding: utf-8 -*-
"""model_NN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aKNPbTHyjOosto3y2r-ztZ7tUOTaJqJM

# Neural Network Binary Classifier TCGA
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
Cancer = pd.read_csv("/content/drive/MyDrive/ALLAB/std_Cancer.txt.bz2",header=None, delimiter = "\t")
Normal = pd.read_csv("/content/drive/MyDrive/ALLAB/std_Normal.txt.bz2",header=None, delimiter = "\t")
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

"""#  Neural Network pan Classifier TCGA"""

# Pan Can Classification Multiclass
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
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
Cancer = pd.read_csv("/content/drive/MyDrive/ALLAB/std_pan_Cancer.txt.bz2",header=None, delimiter = "\t")
Normal = pd.read_csv("/content/drive/MyDrive/ALLAB/std_pan_Normal.txt.bz2",header=None, delimiter = "\t")
Normal = Normal.drop(Normal.index[0])
frame = [Cancer,Normal]
Data = pd.concat(frame,axis=0)
Data = Data.drop(Data.index[0])
x = Data.iloc[:,:300]
Y = Data.iloc[:,300]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(x, dummy_y, test_size = 0.25)

class myCallback(tf.keras.callbacks.Callback): 
  def on_epoch_end(self, epoch, logs={}): 
    if(logs.get('accuracy') > 0.93):   
      #print("\nWe have reached %2.2f%% accuracy, so we will stopping training." %(0.99*100))   
      self.model.stop_training = True

# define baseline model
def baseline_model():
	# create model
  model = Sequential()
  model.add(Dense(64, input_dim=300, activation='relu'))
  model.add(Dense(32, activation='relu')) #hidden layer #1
  model.add(Dense(16, activation='relu')) #hidden layer #2
  model.add(Dense(23, activation='softmax'))
	# Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
callbacks = myCallback()
model = baseline_model()
#model.fit(X_train, y_train, epochs=35, batch_size=100 , callbacks=[callbacks], verbose=1)
model.fit(X_train, y_train, epochs=35, batch_size=100 , verbose=1)
y_pred = model.predict_classes(X_test)

y_test_flat = [np.where(r==1)[0][0] for r in y_test]

print("---------(Neural Network)-----------")
print(confusion_matrix(y_test_flat,y_pred))
print(classification_report(y_test_flat,y_pred))
print("MCC Score (Neural Network): ",matthews_corrcoef(y_test_flat, y_pred))
print("---------(Neural Network)-----------")