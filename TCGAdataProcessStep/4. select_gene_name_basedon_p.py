import numpy as np
import pandas as pd
import csv
import random

# importing the unique m genes name
unique_genes = np.load("/home/vm21/PanClassify/data/genes_that_willbe_filtered.npy")

# reading and making a dictonary of counted numbers of genes based on their frequency
with open('/home/vm21/PanClassify/std_npy/gene_frequency.csv', mode='r') as infile:
    reader = csv.reader(infile)
    gene_freq_dict = {rows[0]:rows[1] for rows in reader}

# print(type(unique_genes))
# print(unique_genes.shape)
# print(len(unique_genes))
# for gense in unique_genes:
# 	print(gense)

genes_with_freq_three = []
selected_genes = []

# selecting genes based on hypothesis one
for gene in unique_genes:
	if(gene_freq_dict[gene] == "3"):
		genes_with_freq_three.append(gene)
	elif(gene_freq_dict[gene] > "3"):
		selected_genes.append(gene)
#print(len(selected_genes))

selected_genes = selected_genes + genes_with_freq_three[0:round(((len(genes_with_freq_three)/100) * 6.57))] #56.92% taken

#print(len(selected_genes))

np.save("/home/vm21/PanClassify/data/selected_genes", selected_genes)
