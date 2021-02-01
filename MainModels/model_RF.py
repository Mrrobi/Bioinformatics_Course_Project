# Oversample and plot imbalanced dataset with SMOTE
import io
import numpy as np
import pandas as pd
import pylab as pl
from scipy import interp
from sklearn import tree
from sklearn.svm import SVC
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
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
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from numpy import where



Cancer = pd.read_csv("/home/vm21/PanClassify/data/std_Cancer.txt.bz2",header=None, delimiter = "\t")
Normal = pd.read_csv("/home/vm21/PanClassify/data/std_Normal.txt.bz2",header=None, delimiter = "\t")


Cancer['Target'] = 1
Normal['Target'] = 0

Normal = Normal.drop(Normal.index[0])

frame = [Cancer,Normal]

Data = pd.concat(frame,axis=0)


Data = Data.drop(Data.index[0])
x = Data.iloc[:,:300]
y = Data.iloc[:,300]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

print("---------(Random Forest)-----------")
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("MCC Score (RF BASIC): ",matthews_corrcoef(y_test, y_pred))
print("---------(Random Forest)-----------")
