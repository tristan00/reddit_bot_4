
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
from collections import OrderedDict
config = configparser.ConfigParser()
import string

paragraph_marker = ' marker_new_paragraph '
new_comment_marker = ' marker_new_comment '
tab_marker = ' marker_new_paragraph '
return_marker = ' marker_return '
db_location = r'C:\Users\tdelforge\Documents\project_dbs\reddit\reddit.db'
display_step = 1000
embedding_size = 128
skip_window = 1
num_skips = 2
num_sampled = 64
valid_size = 16
valid_window = 100
n_input = 200
n_hidden = 512
dnn_n_hidden = 1024
batch_size = 1000
logs_path = '/tmp/tensorflow_logs/example/'
sorted_subreddit_list = []



valid_examples = np.random.choice(valid_window, valid_size, replace=False)
data_index = 0
possible_inputs = list(string.printable)
input_len = len(possible_inputs) + 2
full_input_size = n_input * input_len
training_iters = 1000000
dnn_in_size = n_input*input_len

value_map = OrderedDict()
inv_value_map = OrderedDict()

def update_dnn_size():
    global dnn_in_size
    with sqlite3.connect(db_location) as conn:
        res = conn.execute('select distinct s_id from subreddits').fetchall()
    dnn_in_size += len(res)

def update_mapping():
    global value_map
    global inv_value_map
    for count, i in enumerate(possible_inputs):
        value_map[i] = count
    value_map[new_comment_marker] = input_len - 1

    inv_value_map = {v: k for k, v in value_map.items()}
    print('mapping updated')

def get_embedding_from_mapping(mapping_array):
    embeddings_output = []

    for letter in mapping_array:
        embeddings_output.append(get_letter_embedding(get_mapping_letter(letter[0])))

    output = np.hstack(embeddings_output)
    return output

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


def get_letter_mapping(input_letter):
    return np.array([value_map.get(input_letter, input_len-2)])


def get_mapping_letter(input_mapping):
    return np.array([inv_value_map.get(input_mapping, '_')])


def get_chain_num_mapping(comments):
    embeddings_output = []

    for comment in comments:
        for letter in comment:
            embeddings_output.append(get_letter_mapping(letter))
        embeddings_output.append(get_letter_mapping(new_comment_marker))

    output = np.vstack(embeddings_output)
    return output


def get_text_from_mapping(mapping_list):
    output = []
    for i in mapping_list:
        output.append(get_mapping_letter(i[0]))
    return ''.join([i[0] for i  in output])


def get_subreddit_features(s_id):
    global sorted_subreddit_list

    if len(sorted_subreddit_list) == 0:
        with sqlite3.connect(db_location) as conn:
            res = conn.execute('select distinct s_id from subreddits').fetchall()
        sorted_subreddit_list = sorted([i[0] for i in res])
    subreddit_features = [0 for _ in sorted_subreddit_list]
    subreddit_features[sorted_subreddit_list.index(s_id)] = 1
    return np.array(subreddit_features)


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
            if count > 170000:
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

'''
learning was slow, curren plan for next steps:
each of the next work embeddings will go to a cell, split the input array in the character representation arrays,
lstm network width needs to be the same as the number of input characters.

goal: ~40% accuracy

TODO:
figure out how to include meta data info such as subreddit or datetime info
figure out if increasing width past size of input does anything, if so how?
'''

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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001).minimize(cost)

    saver = tf.train.Saver()
    start_time = time.time()

    with tf.Session() as session:
        step = 0
        counter = 0

        try:
            saver.restore(session, "models/lstm_model")
        except:
            traceback.print_exc()
            session.run(init)
        loss_total = 0
        acc_total = 0

        while step < training_iters:
            next_chain = random.choice(comment_chains)
            #chain_embedding = get_chain_embeddings(next_chain)


            chain_mapping = get_chain_num_mapping(next_chain)
            if chain_mapping.shape[0] - 2 <= n_input:
                continue
            input_index = random.randint(n_input, chain_mapping.shape[0] - 2)

            output_map = chain_mapping[input_index]
            #print(input_index, output_map)
            output_letter = get_mapping_letter(output_map[0])
            output_embedding = get_letter_embedding(output_letter)

            input_mapping = chain_mapping[(input_index - n_input):input_index]

            #input_embeddings = chain_embedding[(input_index-n_input):input_index]
            symbols_in = np.reshape(input_mapping, (-1, n_input, 1))
            #symbols_in = np.reshape(np.array([i.tolist().index(1) for i in input_embeddings]), (-1, n_input, 1))
            #symbols_in = np.reshape(chain_embedding[(input_index-n_input):input_index], [-1, full_input_size, 1])


            symbols_out = np.reshape(output_embedding, [1, -1])
            #symbols_out = np.reshape(chain_embedding[input_index], [1, -1])
            _, acc, loss, pred = session.run([optimizer, accuracy, cost, model], feed_dict={x: symbols_in, y: symbols_out})

            loss_total += loss
            acc_total += acc
            if counter % display_step == 0:
                input_text = get_text_from_mapping(input_mapping)

                correct_letter = get_letter_from_embedding(np.reshape(symbols_out, (input_len,)))
                pred_letter = get_letter_from_embedding(np.reshape(pred, (input_len,)))

                # correct_letter = get_letter_from_embedding(np.reshape(symbols_out, (input_len,)))
                # pred_letter = get_letter_from_embedding(np.reshape(pred, (input_len,)))

                print(step, time.time()- start_time,input_index, acc_total/display_step, loss_total/display_step)
                print('input sentence:', input_text)
                if pred_letter != correct_letter:
                    print('picked {0} instead of {1}:'.format(pred_letter, correct_letter))
                else:
                    print('correctly predicted:', correct_letter)
                print()
                loss_total = 0
                acc_total = 0
            counter += 1
            input_index += 1

            step += 1
        saver.save(session, 'models/lstm_model')


def generate_input(comment_chains, num):
    count = 0

    results = []
    while count < num:
        next_chain = random.choice(comment_chains)
        chain_mapping = get_chain_num_mapping(next_chain)
        if chain_mapping.shape[0] - 2 <= n_input:
            continue
        input_index = random.randint(n_input, chain_mapping.shape[0] - 2)
        output_map = chain_mapping[input_index]
        output_letter = get_mapping_letter(output_map[0])
        output_embedding = get_letter_embedding(output_letter)
        input_mapping = chain_mapping[(input_index - n_input):input_index]
        symbols_in = np.reshape(input_mapping, (-1, n_input, 1))
        symbols_out = np.reshape(output_embedding, [1, -1])
        input_text = get_text_from_mapping(input_mapping)
        correct_letter = get_letter_from_embedding(np.reshape(symbols_out, (input_len,)))

        results.append({'in':symbols_in,
                        'out':symbols_out,
                        'text':input_text,
                        'correct_output_text':correct_letter})

        count += 1
    return results

#btatch inputs
def train_model3():
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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.0001).minimize(cost)

    saver = tf.train.Saver()
    start_time = time.time()


    with tf.Session() as session:
        step = 0
        counter = 0

        try:
            saver.restore(session, "models/lstm_model")
        except:
            traceback.print_exc()
            session.run(init)
        print('model loaded', time.time() - start_time)

        while step < training_iters:
            input_list = generate_input(comment_chains, batch_size)
            symbols_in = np.vstack([i['in'] for i in input_list])
            symbols_out = np.vstack([i['out'] for i in input_list])
            print('input generated', time.time() - start_time)

            _, acc, loss, pred = session.run([optimizer, accuracy, cost, model], feed_dict={x: symbols_in, y: symbols_out})
            pred_letter = get_letter_from_embedding(np.reshape(pred[-1], (input_len,)))
            correct_letter = input_list[-1]['correct_output_text']

            print(step, time.time()- start_time, acc, loss)
            print('input sentence:', input_list[-1]['text'])
            if pred_letter != correct_letter:
                print('picked {0} instead of {1}:'.format(pred_letter, correct_letter))
            else:
                print('correctly predicted:', correct_letter)
            print()

            counter += batch_size

            step += batch_size
        saver.save(session, 'models/lstm_model')


def get_model4(x):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([dnn_in_size, dnn_n_hidden*20])),
                      'biases': tf.Variable(tf.random_normal([dnn_n_hidden*20]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([dnn_n_hidden*20, dnn_n_hidden*10])),
                      'biases': tf.Variable(tf.random_normal([dnn_n_hidden*10]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([dnn_n_hidden*10, dnn_n_hidden*6])),
                      'biases': tf.Variable(tf.random_normal([dnn_n_hidden*6]))}
    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([dnn_n_hidden*6, dnn_n_hidden*3])),
                      'biases': tf.Variable(tf.random_normal([dnn_n_hidden*3]))}
    hidden_5_layer = {'weights': tf.Variable(tf.random_normal([dnn_n_hidden*3, dnn_n_hidden])),
                      'biases': tf.Variable(tf.random_normal([dnn_n_hidden]))}
    out_layer = {'weights': tf.Variable(tf.random_normal([dnn_n_hidden, input_len])),
                      'biases': tf.Variable(tf.random_normal([input_len]))}

    l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.leaky_relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.leaky_relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.leaky_relu(l3)
    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.leaky_relu(l4)
    l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
    l5 = tf.nn.leaky_relu(l5)

    output = tf.add(tf.matmul(l5, out_layer['weights']), out_layer['biases'])
    return output


def generate_input_and_array(comment_chains, num):
    count = 0

    results = []
    while count < num:
        next_chain = random.choice(comment_chains)
        next_chain_text = next_chain['text']
        chain_mapping = get_chain_num_mapping(next_chain_text)
        if chain_mapping.shape[0] - 2 <= n_input:
            continue
        subreddit_features = get_subreddit_features(next_chain['s_id'])
        input_index = random.randint(n_input, chain_mapping.shape[0] - 2)
        output_map = chain_mapping[input_index]
        output_letter = get_mapping_letter(output_map[0])
        output_embedding = get_letter_embedding(output_letter)
        input_mapping = chain_mapping[(input_index - n_input):input_index]
        input_array = get_embedding_from_mapping(input_mapping)
        input_array = np.hstack((input_array, subreddit_features))
        symbols_in = np.reshape(input_mapping, (-1, n_input, 1))
        symbols_out = np.reshape(output_embedding, [1, -1])
        input_text = get_text_from_mapping(input_mapping)
        correct_letter = get_letter_from_embedding(np.reshape(symbols_out, (input_len,)))

        results.append({'in':symbols_in,
                        'out':symbols_out,
                        'text':input_text,
                        'correct_output_text':correct_letter,
                        'array': input_array,
                        'sub_features': subreddit_features})

        count += 1
    return results

def get_data2():
    with sqlite3.connect(db_location) as conn:
        posts = conn.execute('select distinct p_id, title, s_id from posts order by score desc').fetchall()

        comments_df = pd.read_sql('select * from comments', conn)
        comment_chains = []
        for count, (p_id, title, s_id) in enumerate(posts):
            if count > 20000:
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
                    comment_chains.append({'text':comment_chain, 's_id':s_id})
        random.shuffle(comment_chains)

        with open('models/letter_data_input2.plk', 'wb') as infile:
            pickle.dump(comment_chains, infile)
            pass

#testing dnn to compare performance with same inputs
def train_model4():
    with open('models/letter_data_input2.plk', 'rb') as infile:
        comment_chains = pickle.load(infile)

    x = tf.placeholder('float', [None, dnn_in_size])
    y = tf.placeholder('float', [None, input_len])

    model = get_model4(x)
    correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.global_variables_initializer()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.000000001).minimize(cost)
    #optimizer = tf.train.RMSPropOptimizer(.001).minimize(cost)

    saver = tf.train.Saver()
    start_time = time.time()


    with tf.Session() as session:
        step = 0
        counter = 0

        try:
            saver.restore(session, "models/dnn_model")
        except:
            traceback.print_exc()
            session.run(init)

        print('model loaded', time.time() - start_time)

        while step < training_iters:
            input_list = generate_input_and_array(comment_chains, batch_size)
            symbols_in = np.vstack([i['array'] for i in input_list])
            #symbols_in = np.vstack([i['in'] for i in input_list])
            #symbols_in = np.squeeze(symbols_in)
            symbols_out = np.vstack([i['out'] for i in input_list])
            print('input generated', time.time() - start_time)

            _, acc, loss, pred = session.run([optimizer, accuracy, cost, model], feed_dict={x: symbols_in, y: symbols_out})
            pred_letter = get_letter_from_embedding(np.reshape(pred[-1], (input_len,)))
            correct_letter = input_list[-1]['correct_output_text']

            print(step, time.time()- start_time, acc, loss)
            print('input sentence:', input_list[-1]['text'])
            if pred_letter != correct_letter:
                print('picked {0} instead of {1}:'.format(pred_letter, correct_letter))
            else:
                print('correctly predicted:', correct_letter)
            print()

            counter += batch_size

            step += batch_size
        saver.save(session, 'models/dnn_model')


if __name__ == '__main__':
    # get_data2()
    update_mapping()
    update_dnn_size()
    train_model4()