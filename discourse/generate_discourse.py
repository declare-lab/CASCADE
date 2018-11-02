#!/usr/bin/env python
import csv
import gensim
import numpy as np
import os 

input_model_path = "./models/discourse_model"
output_folder = "./discourse_features"

print('loading topics model...')
model = gensim.models.Doc2Vec.load(input_model_path)
print('topics model loaded')


# for topic embeddings
topic_embeddings = []
topic_ids = model.docvecs.offset2doctag
topic_embeddings_size = 100
for k in topic_ids:
    try:
        lst = model.docvecs[k]
    except TypeError:
        lst = np.random.normal(size=topic_embeddings_size)
    topic_embeddings.append(lst)
    
topic_ids = [0] + topic_ids
unknown_vector = np.random.normal(size=(1, topic_embeddings_size))
topic_embeddings = np.concatenate((unknown_vector, topic_embeddings), axis=0)
topic_embeddings = topic_embeddings.astype(dtype='float32')

print("len of topic embeddings: ", len(topic_embeddings))

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
with open(output_folder+"/discourse.csv", "w") as fp:
    csv_writer = csv.writer(fp)
    for i in range(len(topic_embeddings)):
        csv_writer.writerow([topic_ids[i], topic_embeddings[i]])
