import numpy as np
import json
import operator
import pandas as pd

gene_freq_count_list = np.load("/home/vm21/PanClassify/std_npy/gene_frequency.npy")
unique_elements, counts_elements = np.unique(gene_freq_count_list, return_counts=True)
print("Unique genes:",len(unique_elements))

top_gene = {}

for i in range(len(unique_elements)):
	top_gene[unique_elements[i]] = counts_elements[i]
sorted_top_gene = dict(sorted(top_gene.items(), key=lambda item: item[1],reverse=True))
import csv
with open('/home/vm21/PanClassify/std_npy/gene_frequency.csv', 'w') as f:
    for key in sorted_top_gene.keys():
        f.write("%s,%d\n"%(key,sorted_top_gene[key]))


np.save("/home/vm21/PanClassify/data/genes_that_willbe_filtered",unique_elements)
data = pd.read_csv("/home/vm21/PanClassify/std_npy/gene_frequency.csv",header=None)
counts = data.groupby(1).count()
counts.to_csv(r'/home/vm21/PanClassify/std_npy/gene_frequency_across_cancer.csv',header=None,index=None,index_label=None)