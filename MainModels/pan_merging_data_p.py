import numpy as np
import pandas as pd

cancer_names = ["BLCA","BRCA","CESC","CHOL","COAD","ESCA","HNSC","KICH","KIRC","KIRP","LIHC","LUAD","LUSC","PAAD","PCPG","PRAD","READ","SARC","STAD","THCA","THYM","UCEC"]
normal_names = ["BLCA","BRCA","CESC","CHOL","COAD","ESCA","HNSC","KICH","KIRC","KIRP","LIHC","LUAD","LUSC","PAAD","PCPG","PRAD","READ","SARC","STAD","THCA","THYM","UCEC"]





#reading data and doing work
cresult=pd.DataFrame()
nresult=pd.DataFrame()
for index in range(len(cancer_names)):
  Cancer = pd.read_csv("/home/vm21/PanClassify/data/std_datas_after_filter/cancer/"+cancer_names[index]+".txt.bz2",header=None, delimiter = "\t")
  Normal = pd.read_csv("/home/vm21/PanClassify/data/std_datas_after_filter/normal/"+normal_names[index]+".norm.txt.bz2",header=None, delimiter = "\t")
  
  Cancer= Cancer.T
  Normal=Normal.T
  Cancer['target'] = cancer_names[index]
  Normal['target'] = "normal"

  frames1 = [Cancer, cresult]
  cresult = pd.concat(frames1)
  
  frames2 = [Normal, nresult]
  nresult = pd.concat(frames2)
  
  print(cresult.shape)
  print(nresult.shape)
  

# merging all the cancer and normal data together separately and saving them  
cresult.to_csv(r'/home/vm21/PanClassify/data/std_pan_Cancer.txt.bz2',compression="bz2", sep='\t',header=None,index=None,index_label=None)

nresult.to_csv(r'/home/vm21/PanClassify/data/std_pan_Normal.txt.bz2',compression="bz2", sep='\t',header=None,index=None,index_label=None)
  
  

  
