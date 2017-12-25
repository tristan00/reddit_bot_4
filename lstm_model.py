import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
import nltk
import re
import sqlite3
import pandas as pd
import nltk
import operator
import functools
import collections

learning_rate = 0.001
training_iters = 100000
display_step = 1000
n_input = 8
n_hidden = 1024
max_vocab_size = 50000
pos_size = 36
total_vocab_size =  max_vocab_size + pos_size
db_location = r'C:\Users\tdelforge\Documents\project_dbs\reddit\reddit.db'
comment_max_len = 100
paragraph_marker = ' marker_new_paragraph '
new_comment_marker = ' marker_new_comment '
tab_marker = ' marker_new_paragraph '
return_marker = ' marker_return '

def get_model(weights, biases):
    x = tf.placeholder("float", [None, n_input, 1])
    y = tf.placeholder("float", [None, total_vocab_size])
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def train_model():
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, total_vocab_size]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([total_vocab_size]))
    }
    x = tf.placeholder("float", [None, n_input, 1])
    y = tf.placeholder("float", [None, total_vocab_size])
    model = get_model(x, weights, biases)

def get_word_features(word_map, word_list):
    return [word_map.get(i, len(word_map.keys())) for i in word_list]

def get_features_from_inputs(tokenized_lists):
    full_list = functools.reduce(operator.concat, tokenized_lists)
    #freq = nltk.FreqDist(full_list)
    counted_list = collections.Counter(full_list).most_common(100)
    word_id_map = {}
    for count, i in enumerate(counted_list):
        word_id_map[i[0]] = count

    for i in tokenized_lists:
        get_word_features(i)
        print(i)
        print()

if __