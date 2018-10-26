#!/usr/bin/env python
import logging

import pandas as pd
import numpy as np
import gensim, os

TaggedDocument = gensim.models.doc2vec.TaggedDocument

#Input file path
USER_PARAGRAPH_INPUTS = "./train_balanced_user"




class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        count = 0
        for idx, doc in enumerate(self.doc_list):
        	try:
	        	yield TaggedDocument(doc.split(), [self.labels_list[idx]])
	        except AttributeError:
	        	count += 1



def trainDoc2Vec(dataset=None):
	''' 
	'''''''''''''''''''''''''''''''''''''''''''''''''
	Doc2Vec training function, trained model is saved
	'''''''''''''''''''''''''''''''''''''''''''''''''
	'''
	assert dataset
	
	data = np.asarray(pd.read_csv(dataset+".csv", header=None))
	docLabels = [ data[i][0] for i in range(data.shape[0])]
	docs = [data[i][1] for i in range(data.shape[0])] 
	it = LabeledLineSentence(docs, docLabels)

	model = gensim.models.Doc2Vec(vector_size=100, window=10, min_count=5, workers=11, alpha=0.025, min_alpha=0.025)
	model.build_vocab(it)
	for epoch in range(50):
		print("Epoch: ", epoch)
		model.alpha -= 0.002
		model.min_alpha = model.alpha
		model.train(it, total_examples=model.corpus_count, epochs=model.iter)
		directory = "./models"
		if not os.path.exists(directory):
			os.makedirs(directory)
		model.save(directory+"/user_stylometric.model")
	

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
trainDoc2Vec(USER_PARAGRAPH_INPUTS)
