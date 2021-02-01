import numpy as np
import pandas as pd
import csv
import random

cancer_names = ["BLCA","BRCA","CESC","CHOL","COAD","ESCA","HNSC","KICH","KIRC","KIRP","LIHC","LUAD","LUSC","PAAD","PCPG","PRAD","READ","SARC","STAD","THCA","THYM","UCEC"]

selected_genes = np.load("/home/vm21/PanClassify/data/selected_genes.npy")
data = pd.read_csv("/home/vm21/PanClassify/cancer/KICH.csv.gz",header=None)
header_gene_names =  data.iloc[:,0:1]
header_gene_names = header_gene_names.drop(header_gene_names.index[0])
header_gene_names.reset_index(drop=True, inplace=True)
# print(header_gene_names)
# print(header_gene_names.shape)

data_read = pd.read_csv("/home/vm21/PanClassify/std_cancer/KICH.txt.bz2",header=None, delimiter = '\t')
# print(data_read)
#print(header_gene_names.shape)
#print(data_read.shape)

# adding gene names in a std cancer data

frame0 = [data_read,header_gene_names]
data_cancer_early = pd.concat(frame0, axis = 1)
data_cancer_early = data_cancer_early.T
data_cancer_early.reset_index(drop=True, inplace=True)

#print(data_cancer_early)

cols_exist = []
cols_to_del = []

#print(data_cancer_early[0][91])

# selecting the genes index that we need from 20501 genes using "selected genes"
for i in range(20501):
	for j in range(len(selected_genes)):
		if(data_cancer_early[i][91] == selected_genes[j]):
			cols_exist.append(i)

# selecting the genes index that we need to drop from "cols_to_exist"
for i in range(20501):
	if(i not in cols_exist):
		cols_to_del.append(i)

# saving the genes that we need to drop
np.save("/home/vm21/PanClassify/data/genes_to_del_300_n",cols_to_del)



	

