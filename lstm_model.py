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

def get_input_text_from_comment_chain(comment_chain):
    formatted_chain = []
    for i in comment_chain:
        formatted_chain.append(i.replace(r'\n', paragraph_marker).replace(r'\t', tab_marker).replace(r'\r',return_marker))
    text_line = new_comment_marker.join(formatted_chain) + new_comment_marker
    return nltk.word_tokenize(text_line)

def get_data(num_of_comment_roots = 10):
    with sqlite3.connect(db_location) as conn:
        # get root comments
        p_df = pd.read_sql('select * from comments where parent_id is Null', conn)

        comment_chains = []
        for count, (i, j) in enumerate(p_df.iterrows()):
            if count > num_of_comment_roots:
                break
            comment_chain = []

            #get parent title
            title_text = conn.execute('select title from posts where p_id = ? limit 1', (j['p_id'], )).fetchone()[0]
            comment_chain.append(title_text)

            #add root comment
            comment_chain.append(j['body'])

            #for now pick highest score child
            parent_id = j['c_id']
            while True:
                next_child = conn.execute('select body, c_id from comments where parent_id = ? order by score DESC', (parent_id,)).fetchone()
                if next_child:
                    reply_text, parent_id = next_child
                    comment_chain.append(reply_text)
                else:
                    break
            comment_chains.append(comment_chain)

    tokenized_inputs = []
    for i in comment_chains:
        tokenized_inputs.append(get_input_text_from_comment_chain(i))
    get_features_from_inputs()




get_data()