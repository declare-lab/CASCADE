import numpy as np
import csv
import pandas as pd
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
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
    return string.strip().lower()


def load_data_and_labels(train_data_file_1, train_data_file_2):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    train_data = np.asarray(pd.read_csv(train_data_file_1, header=None, encoding="cp1252"))
    users = train_data[:,0]
    paragraphs = train_data[:,1]
    Labels = []
    for i in range(len(train_data)):
        label = []
        for j in range(2,7,1):
            if(train_data[i][j] == 'y'):
                label.append(1)
            else:
                label.append(0)
        Labels.append(label)

    paragraphs = [x.strip() for x in paragraphs]
    paragraphs = [clean_str(x) for x in paragraphs]
    paragraphs = np.asarray(paragraphs)
    Labels = np.asarray(Labels)
    print(paragraphs.shape)
    print(Labels.shape)
    fb_train_data = np.asarray(pd.read_csv(train_data_file_2, header=None, encoding="cp1252"))
    fb_paragraphs = fb_train_data[:,0]
    fb_Labels = []
    for i in range(len(fb_train_data)):
        fb_Labels.append([int(fb_train_data[i][j]) for j in range(1,6,1)])
    fb_paragraphs = [x.strip() for x in fb_paragraphs]
    fb_paragraphs = [clean_str(x) for x in fb_paragraphs]
    fb_paragraphs = np.asarray(fb_paragraphs)
    fb_Labels = np.asarray(fb_Labels)
    paragraphs = np.append(paragraphs,fb_paragraphs)
    Labels = np.append(Labels,fb_Labels,0)
    print(fb_paragraphs.shape)
    print(fb_Labels.shape)
    print(paragraphs.shape)
    print(Labels.shape)

    return [paragraphs,Labels]

def load_data_and_labels_test(test_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    test_data = np.asarray(pd.read_csv(test_data_file, header=None,sep = '\t'))
    users = test_data[:,2]
    paragraphs = test_data[:,1]
    # Labels = []
    # for i in range(len(test_data)):
    #     Labels.append([int(test_data[i][j]) for j in range(1,6,1)])
    paragraphs = [(x.strip())[0:1000] for x in paragraphs if str(x) != 'nan']
    paragraphs = [clean_str(x) for x in paragraphs]
    paragraphs = np.asarray(paragraphs)
    # Labels = np.asarray(Labels)
    return [users, paragraphs]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
	print "epoch: ",epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
