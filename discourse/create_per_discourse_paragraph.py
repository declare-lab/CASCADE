#!/usr/bin/env python
import csv
import numpy as np
import pandas as pd

data = np.asarray(pd.read_csv("./../data/train-balanced.csv", header=None, sep='\t'))
topics = set(data[:, 3])
print(len(topics))

file = open("./train_balanced_topics.csv", 'w')
max_len = 0
wr = csv.writer(file, quoting=csv.QUOTE_ALL)

count = 0
for i in range(len(data)):
    if str(data[i][1]) != 'nan' and len(data[i][1].split()) > 500:
        count += 1
    if str(data[i][1]) != 'nan' and max_len < len(data[i][1].split()):
        max_len = len(data[i][1].split())
new_data = []

for ind, topic in enumerate(topics):
    ls = []
    comments = data[data[:, 3] == topic, 1]
    comments = [x for x in comments if str(x) != 'nan']

    comment = " <END> ".join(comments)
    ls.append(topic)
    ls.append(comment)
    wr.writerow(ls)
