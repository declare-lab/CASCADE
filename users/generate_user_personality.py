#!/usr/bin/env python
import tensorflow as tf
import pickle
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import sys

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("test_data_file", "./../data/train-balanced.csv", "Data source")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

print("loading data...",)
users, x_raw = data_helpers.load_data_and_labels_test(FLAGS.test_data_file)
x = pickle.load(open("mr.p","rb"))
revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
print("data loaded!")


x = []
for i in range(len(x_raw)):
    lst = []
    try:
        for word in x_raw[i].split():
            if word in word_idx_map:
                lst.append(word_idx_map[word])
            else:
                lst.append(0)
    except AttributeError:
        lst.append(0)
    x.append(lst)

for i in range(len(x)):
        if( len(x[i]) < 1000 ):
                x[i] = np.append(x[i],np.zeros(1000-len(x[i])))
        elif( len(x[i]) > 1000 ):
                x[i] = x[i][0:1000]
x_test = np.asarray(x)
num_filters = 128

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph('./my_model-1.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        h_last = graph.get_tensor_by_name("last_layer/h_last:0")
        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        csvfile = "./user_embeddings/user_personality.csv"
        
        user_set = set(users)
    mp_vec = {}
    mp_count = {}
    for user in user_set:
        mp_vec[user] = [0]*100
        mp_count[user] = 0

    print(len(user_set))
    all_h = []
    count = 0
    
    user_ind = 0
    for x_test_batch in batches:

        batch_h = sess.run(h_last, {input_x: x_test_batch, dropout_keep_prob: 1.0})
        for i in range(len(x_test_batch)):
            mp_vec[users[user_ind]] += batch_h[i]
            mp_count[users[user_ind]] += 1
            user_ind += 1

    res = []
    for user in user_set:
        print(user)
        ls = []
        if mp_count[user] == 0:
            print("Error")
            exit()
        
        for i in range(100):
            mp_vec[user][i] /= mp_count[user]
        
        print(user, mp_vec[user])
        ls.append(user)
        ls.append(mp_vec[user])
        res.append(ls)

    with open(csvfile, "w") as output:
        writer = csv.writer(output)
        for val in res:
            writer.writerow(val)
