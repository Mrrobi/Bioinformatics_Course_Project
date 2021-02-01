import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

cancer_names = ["BLCA","BRCA","CESC","CHOL","COAD","ESCA","HNSC","KICH","KIRC","KIRP","LIHC","LUAD","LUSC","PAAD","PCPG","PRAD","READ","SARC","STAD","THCA","THYM","UCEC"]
normal_names = ["BLCA","BRCA","CESC","CHOL","COAD","ESCA","HNSC","KICH","KIRC","KIRP","LIHC","LUAD","LUSC","PAAD","PCPG","PRAD","READ","SARC","STAD","THCA","THYM","UCEC"]

#reading data and doing work
for index in range(len(cancer_names)):
  Cancer = pd.read_csv("/home/vm21/PanClassify/cancer/"+cancer_names[index]+".csv.gz",header=None)
  Normal = pd.read_csv("/home/vm21/PanClassify/normal/"+normal_names[index]+".norm.csv.gz",header=None)

  #droping sample names needed man whats your problem?
  Cancer = Cancer.drop(Cancer.index[0])
  Cancer = Cancer.drop(columns=[0])
  Normal = Normal.drop(Normal.index[0])
  Normal = Normal.drop(columns=[0])

  # create a scaler object
  std_scaler_can = StandardScaler()
  std_scaler_norm = StandardScaler()
  # fit and transform the data
  df_std_can = pd.DataFrame(std_scaler_can.fit_transform(Cancer), columns=Cancer.columns)
  df_std_norm = pd.DataFrame(std_scaler_norm.fit_transform(Normal), columns=Normal.columns)
  df_std_can.to_csv(r'/home/vm21/PanClassify/std_cancer/'+cancer_names[index]+".txt.bz2",compression="bz2", sep='\t',header=None,index=None,index_label=None)
  df_std_norm.to_csv(r'/home/vm21/PanClassify/std_norm/'+cancer_names[index]+".txt.bz2",compression="bz2", sep='\t',header=None,index=None,index_label=None)

