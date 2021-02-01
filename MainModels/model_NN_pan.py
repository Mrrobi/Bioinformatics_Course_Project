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
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
Cancer = pd.read_csv("/content/drive/MyDrive/ALLAB/std_pan_Cancer.txt.bz2",header=None, delimiter = "\t")
Normal = pd.read_csv("/content/drive/MyDrive/ALLAB/std_pan_Normal.txt.bz2",header=None, delimiter = "\t")
Cancer['Target'] = 1
Normal['Target'] = 0
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

# define baseline model
def baseline_model():
	# create model
  model = Sequential()
  model.add(Dense(64, input_dim=300, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(23, activation='softmax'))
	# Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

model = baseline_model()
model.fit(X_train, y_train, epochs=25, batch_size=100)

y_pred = model.predict_classes(X_test)

y_test_flat = [np.where(r==1)[0][0] for r in y_test]

print("---------(Neural Network)-----------")
print(confusion_matrix(y_test_flat,y_pred))
print(classification_report(y_test_flat,y_pred))
print("MCC Score (Neural Network): ",matthews_corrcoef(y_test_flat, y_pred))
print("---------(Neural Network)-----------")