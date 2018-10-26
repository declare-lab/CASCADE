#!/usr/bin/env python
import pandas as pd
import numpy as np
import csv

doc_data = np.asarray(pd.read_csv('./user_embeddings/user_stylometric.csv', header=None))
per_data = np.asarray(pd.read_csv('./user_embeddings/user_personality.csv', header=None))
csvfile = open("./user_embeddings/user_view_vectors.csv",'w')
wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

x = [doc_data[i][1].split(",") for i in range(len(doc_data))]
for i in range(len(doc_data)):
	x[i][0] = x[i][0][1:]
	x[i][len(x[i])-1] = x[i][len(x[i])-1][:-1]

y = [per_data[i][1].split(",") for i in range(len(per_data))]
for i in range(len(per_data)):
	y[i][0] = y[i][0][1:]
	y[i][len(y[i])-1] = y[i][len(y[i])-1][:-1]

x = np.asarray(x)
y = np.asarray(y)

map_dv = {}
for i in range(len(doc_data)):
	map_dv[doc_data[i][0]] = x[i]

map_pv = {}
for i in range(len(per_data)):
	map_pv[per_data[i][0]] = y[i]

users = doc_data[:,0]
for user in users:
	ls = []
	ls.append(user)
	ls.append(float(len(map_dv[user])))
	ls.append(float(len(map_pv[user])))
	for j in range(len(map_dv[user])):
		ls.append(float(map_dv[user][j]))
	for j in range(len(map_pv[user])):
		ls.append(float(map_pv[user][j]))
	wr.writerow(ls)
