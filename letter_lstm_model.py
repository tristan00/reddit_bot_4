
import numpy as np
import configparser
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
import pickle
import math
from sklearn.manifold import TSNE
import traceback
import os
from tempfile import gettempdir
import matplotlib.pyplot as plt
config = configparser.ConfigParser()
import string

paragraph_marker = ' marker_new_paragraph '
new_comment_marker = ' marker_new_comment '
tab_marker = ' marker_new_paragraph '
return_marker = ' marker_return '
db_location = r'C:\Users\tdelforge\Documents\project_dbs\reddit\reddit.db'
display_step = 100
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
num_sampled = 64
valid_size = 16
valid_window = 100
n_input = 200
n_hidden = 512
logs_path = '/tmp/tensorflow_logs/example/'


valid_examples = np.random.choice(valid_window, valid_size, replace=False)
data_index = 0
possible_inputs = list(string.printable)
input_len = len(possible_inputs) + 2
full_input_size = n_input * input_len
training_iters = 25


def get_letter_from_embedding(embedding_array):
    in_list = embedding_array.tolist()
    res_index = in_list.index(max(in_list))

    if res_index < len(possible_inputs):
        return possible_inputs[res_index]
    elif res_index == len(possible_inputs):
        return '_'
    else:
        return new_comment_marker


def get_chain_from_embeddings(embedding_array):
    output = []
    for j in embedding_array:
        output.append(get_letter_from_embedding(j))
    return ''.join(output)


def get_letter_embedding(input_letter):
    output = [0 for _ in range(input_len)]
    if input_letter in possible_inputs:
        output[possible_inputs.index(input_letter)] = 1
    elif len(input_letter) == 1:
        output[-2] = 1
    else:
        output[-1] = 1
    return np.array(output)


def get_chain_embeddings(comments):
    embeddings_output = []

    for comment in comments:
        for letter in comment:
            embeddings_output.append(get_letter_embedding(letter))
        embeddings_output.append(get_letter_embedding(new_comment_marker))

    output = np.vstack(embeddings_output)
    return output


def get_child_list(df, c_id):
    child_df = df[df['parent_id'] == c_id]
    child_df = child_df.sort_values(by='score', ascending=False)
    if child_df.shape[0] == 0:
        return []
    else:
        return [child_df['body'].iloc[0]] + get_child_list(df, child_df['c_id'].iloc[0])


def get_data():
    with sqlite3.connect(db_location) as conn:
        posts = conn.execute('select p_id, title from posts order by score desc').fetchall()

        comments_df = pd.read_sql('select * from comments', conn)
        comment_chains = []
        for count, (p_id, title) in enumerate(posts):
            if count > 25000:
                break
            print(count, len(comment_chains), time.time())
            #df = pd.read_sql('select * from comments where p_id = ?', conn, params=(p_id,))
            post_comments_df = comments_df[comments_df['p_id'] == p_id]
            roots = post_comments_df[post_comments_df['parent_id'].isnull()]
            for _, root in roots.iterrows():
                comment_chain = []
                comment_chain.append(title)
                comment_chain.append(root['body'])
                child_list = get_child_list(post_comments_df, root['c_id'])
                if len(child_list) > 0:
                    comment_chain.extend(child_list)
                    comment_chains.append(comment_chain)
        random.shuffle(comment_chains)

        with open('models/letter_data_input.plk', 'wb') as infile:
            pickle.dump(comment_chains, infile)
            pass


def get_model(x, weights, biases):
    x = tf.reshape(x, [-1, full_input_size])
    x = tf.split(x, n_input, 1)
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def train_model():

    with open('models/letter_data_input.plk', 'rb') as infile:
        comment_chains = pickle.load(infile)

    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, input_len]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([input_len]))
    }
    x = tf.placeholder("float", [None, full_input_size, 1])
    y = tf.placeholder("float", [None, input_len])
    model = get_model(x, weights, biases)

    correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001).minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as session:
        try:
            saver.restore(session, "models/lstm_model")
        except:
            traceback.print_exc()
            session.run(init)
        step = 0
        counter = 0
        while step < training_iters:
            next_chain = random.choice(comment_chains)
            chain_embedding = get_chain_embeddings(next_chain)
            input_index = n_input
            loss_total = 0
            acc_total = 0

            while input_index < chain_embedding.shape[0] - 1:
                symbols_in = np.reshape(chain_embedding[(input_index-n_input):input_index], [-1, full_input_size, 1])
                symbols_out = np.reshape(chain_embedding[input_index], [1, -1])
                _, acc, loss, pred = session.run([optimizer, accuracy, cost, model], feed_dict={x: symbols_in, y: symbols_out})

                loss_total += loss
                acc_total += acc
                if counter%display_step == 0 and counter > 0:
                    correct_letter = get_letter_from_embedding(np.reshape(symbols_out, (input_len,)))
                    pred_letter = get_letter_from_embedding(np.reshape(pred, (input_len,)))

                    print(step, input_index, acc_total/display_step, loss_total/display_step)
                    print('input sentence:', get_chain_from_embeddings(chain_embedding[input_index-n_input:input_index]))
                    print('picked {0} instead of {1}:'.format(pred_letter, correct_letter))
                    print()
                    loss_total = 0
                    acc_total = 0
                    break
                counter += 1
                input_index += 1

            step += 1
        saver.save(session, 'models/lstm_model')

def get_model2(x, weights, biases):
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_input, 1)
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

#does not take input as embedding
def train_model2():

    with open('models/letter_data_input.plk', 'rb') as infile:
        comment_chains = pickle.load(infile)

    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, input_len]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([input_len]))
    }
    x = tf.placeholder("float", [None, n_input, 1])
    y = tf.placeholder("float", [None, input_len])


    model = get_model2(x, weights, biases)
    correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.global_variables_initializer()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01).minimize(cost)

    saver = tf.train.Saver()
    with tf.Session() as session:
        step = 0
        counter = 0

        try:
            saver.restore(session, "models/lstm_model")
        except:
            traceback.print_exc()
            session.run(init)


        while step < training_iters:
            next_chain = random.choice(comment_chains)
            chain_embedding = get_chain_embeddings(next_chain)
            input_index = n_input
            loss_total = 0
            acc_total = 0

            while input_index < chain_embedding.shape[0] - 1:
                input_embeddings = chain_embedding[(input_index-n_input):input_index]
                symbols_in = np.reshape(np.array([i.tolist().index(1) for i in input_embeddings]), (-1, n_input, 1))
                #symbols_in = np.reshape(chain_embedding[(input_index-n_input):input_index], [-1, full_input_size, 1])

                symbols_out = np.reshape(chain_embedding[input_index], [1, -1])
                _, acc, loss, pred = session.run([optimizer, accuracy, cost, model], feed_dict={x: symbols_in, y: symbols_out})


                loss_total += loss
                acc_total += acc
                if counter%displa+-*_step == 0:
                    correct_letter = get_letter_from_embedding(np.reshape(symbols_out, (input_len,)))
                    pred_letter = get_letter_from_embedding(np.reshape(pred, (input_len,)))

                    print(step, input_index, acc_total/display_step, loss_total/display_step)
                    print('input sentence:', get_chain_from_embeddings(chain_embedding[input_index-n_input:input_index]))
                    print('picked {0} instead of {1}:'.format(pred_letter, correct_letter))
                    print()
                    loss_total = 0
                    acc_total = 0
                counter += 1
                input_index += 1

            step += 1
        saver.save(session, 'models/lstm_model')

if __name__ == '__main__':
    #get_data()
    train_model2()