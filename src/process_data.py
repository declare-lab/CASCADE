#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd
import json

# noinspection PyCompatibility
from builtins import range

COMMENTS_FILE = "../data/comments.json"
TRAIN_MAP_FILE = "../data/my_train_balanced.csv"
TEST_MAP_FILE = "../data/my_test_balanced.csv"

def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data
    """
    revs = []

    sarc_train_file = data_folder[0]
    sarc_test_file = data_folder[1]
    
    train_data = np.asarray(pd.read_csv(sarc_train_file, header=None))
    test_data = np.asarray(pd.read_csv(sarc_test_file, header=None))

    comments = json.loads(open(COMMENTS_FILE).read())
    vocab = defaultdict(float)



    
    for line in train_data: 
        rev = []
        label_str = line[2]
        if( label_str == 0):
            label = [1, 0]
        else:
            label = [0, 1]
        rev.append(comments[line[0]]['text'].strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        orig_rev = (orig_rev.split())[0:100]
        orig_rev = " ".join(orig_rev)
        datum  = {"y":int(1), 
                  "id":line[0],
                  "text": orig_rev,
                  "author": comments[line[0]]['author'],
                  "topic": comments[line[0]]['subreddit'],
                  "label": label,
                  "num_words": len(orig_rev.split()),
                  "split": int(1)}
        revs.append(datum)
        
    for line in test_data:
        rev = []
        label_str = line[2]
        if( label_str == 0):
            label = [1, 0]
        else:
            label = [0, 1]
        rev.append(comments[line[0]]['text'].strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        orig_rev = (orig_rev.split())[0:100]
        orig_rev = " ".join(orig_rev)
        datum  = {"y":int(1),
                  "id": line[0], 
                  "text": orig_rev,  
                  "author": comments[line[0]]['author'],
                  "topic": comments[line[0]]['subreddit'],
                  "label": label,
                  "num_words": len(orig_rev.split()),                      
                  "split": int(0)}
        revs.append(datum)
        

    return revs, vocab

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def loadGloveModel(gloveFile, vocab):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        if word in vocab:
               model[word] = embedding

    print("Done.",len(model)," words loaded!")
    return model

def load_fasttext(fname, vocab):
    """
    Loads 300x1 word vecs from Fasttext
    """
    print("Loading FastText Model")
    f = open(fname,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        if word in vocab:
               model[word] = embedding

    print("Done.", len(model), " words loaded!")
    return model

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":  

    w2v_file = sys.argv[1]    
    data_folder = [TRAIN_MAP_FILE,TEST_MAP_FILE] 
    print("loading data...")
    revs, vocab = build_data_cv(data_folder,  cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    print("loading word2vec vectors...")
    w2v = load_fasttext(w2v_file, vocab) 
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    pickle.dump([revs, W, W2, word_idx_map, vocab, max_l], open("mainbalancedpickle.p", "wb"))
    print("dataset created!")
