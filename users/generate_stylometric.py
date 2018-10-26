#!/usr/bin/env python
import pandas as pd
import numpy as np
import csv
import gensim, os

doc2vec = gensim.models.Doc2Vec.load('./models/user_stylometric.model')
data = np.asarray(pd.read_csv('./train_balanced_user.csv', header=None))
DIM = 300

directory = "./user_embeddings"
if not os.path.exists(directory):
	os.makedirs(directory)
file = open(directory+"/user_stylometric.csv",'w')
wr = csv.writer(file, quoting=csv.QUOTE_ALL)

# Inferring paragraphVec vectors for each user
vectors = np.asarray([doc2vec.infer_vector(data[i][1]) for i in range(data.shape[0])])

users = data[:,0]	
for i in range(len(users)):
	ls=[]
	ls.append(users[i])
	v = [0]*100
	for j in range(len(vectors[i])):
		v[j] = vectors[i][j]
	ls.append(v)
	wr.writerow(ls)
