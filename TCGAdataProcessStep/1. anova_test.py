import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2 , f_classif,f_regression,mutual_info_classif,mutual_info_regression
from sklearn.svm import SVR


cancer_names = ["BLCA","BRCA","CESC","CHOL","COAD","ESCA","HNSC","KICH","KIRC","KIRP","LIHC","LUAD","LUSC","PAAD","PCPG","PRAD","READ","SARC","STAD","THCA","THYM","UCEC"]
normal_names = ["BLCA","BRCA","CESC","CHOL","COAD","ESCA","HNSC","KICH","KIRC","KIRP","LIHC","LUAD","LUSC","PAAD","PCPG","PRAD","READ","SARC","STAD","THCA","THYM","UCEC"]

#reading data and doing work
for index in range(len(cancer_names)):
  Cancer = pd.read_csv("/home/vm21/PanClassify/std_cancer/"+cancer_names[index]+".txt.bz2",delimiter = "\t",header=None)
  Normal = pd.read_csv("/home/vm21/PanClassify/std_norm/"+normal_names[index]+".txt.bz2",delimiter = "\t",header=None) 

  #transpose
  Cancer_T = Cancer.T
  Normal_T = Normal.T

  #setting target
  Cancer_T["target"] = 1.0
  Normal_T["target"] = 0.0
  #concating
  X = pd.concat((Cancer_T,Normal_T),axis=0)
  x = X.iloc[:,:20501]
  y = X.iloc[:,20501]

  #selecting k value
  k =  300

  #Anova test
  selector = SelectKBest(f_classif, k=k)
  selector.fit(x, y)
  cols_anova = selector.get_support(indices=True)
  np.save("/home/vm21/PanClassify/std_npy/"+cancer_names[index],cols_anova)