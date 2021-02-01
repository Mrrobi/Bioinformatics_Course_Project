import numpy as np
import pandas as pd
import csv
import random

cancer_names = ["BLCA","BRCA","CESC","CHOL","COAD","ESCA","HNSC","KICH","KIRC","KIRP","LIHC","LUAD","LUSC","PAAD","PCPG","PRAD","READ","SARC","STAD","THCA","THYM","UCEC"]

# loading the genes that need to delete and saving the data again after removing the deleted genes
selected_genes = np.load("/home/vm21/PanClassify/data/genes_to_del_300_n.npy")


for index in range(len(cancer_names)):
	Cancer = pd.read_csv("/home/vm21/PanClassify/std_cancer/"+cancer_names[index]+".txt.bz2",header=None, delimiter = '\t')
	Normal = pd.read_csv("/home/vm21/PanClassify/std_norm/"+cancer_names[index]+".txt.bz2",header=None, delimiter = '\t')
	
	Cancer = Cancer.T
	Normal = Normal.T

	# removing the genes
	cancer = Cancer.drop(columns = selected_genes)
	normal = Normal.drop(columns = selected_genes)

	print(cancer.shape)
	print(normal.shape)

	cancer = cancer.T
	normal = normal.T

	cancer.to_csv(r'/home/vm21/PanClassify/data/std_datas_after_filter/cancer/'+cancer_names[index]+'.txt.bz2',compression="bz2", sep='\t',header=None,index=None,index_label=None)
	normal.to_csv(r'/home/vm21/PanClassify/data/std_datas_after_filter/normal/'+cancer_names[index]+'.norm.txt.bz2',compression="bz2", sep='\t',header=None,index=None,index_label=None)

	

