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
import pickle
import nltk
import operator
import functools
import collections
from scipy.spatial import distance, minkowski_distance

db_location = r'C:\Users\tdelforge\Documents\project_dbs\reddit\reddit.db'
comment_max_len = 100
paragraph_marker = ' marker_new_paragraph '
new_comment_marker = ' marker_new_comment '
tab_marker = ' marker_new_paragraph '
return_marker = ' marker_return '
count, data, dictionary, reverse_dictionary, final_embeddings = None, None, None, None, None

n_input = 50
n_hidden = 512
training_iters = 10000
learning_rate = 0.001
display_step = 10
vocab_size = 128
full_input_size = n_input*vocab_size





def get_training_data(data, dictionary, embeddings_df):
    training_data = []
    for i in data[0:100]:
        training_data.append(embeddings_df.loc[i].values)
    print(1)
    #np_training_data = np.array(training_data)
    return training_data

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

def load_models():
    with open('models2/count.plk', 'rb') as f:
        count = pickle.load(f)
    with open('models2/data.plk', 'rb') as f:
        data = pickle.load(f)
    with open('models2/dictionary.plk', 'rb') as f:
        dictionary = pickle.load(f)
    with open('models2/reverse_dictionary.plk', 'rb') as f:
        reverse_dictionary = pickle.load(f)
    with open('models2/final_embeddings.plk', 'rb') as f:
        final_embeddings = pickle.load(f)
    return count, data, dictionary, reverse_dictionary, final_embeddings

def get_word_similarity(word_index, test_word_index, embeddings_df):
    difference = 0
    word_row = embeddings_df.iloc[word_index].values
    test_word_row = embeddings_df.iloc[test_word_index].values

    for i, j in zip(word_row, test_word_row):
        difference += (i - j)**2
    return difference

def get_closest_word_embeddings_df(input_word, embeddings_df, dictionary, num_of_words):
    word_mapping = dictionary.get(input_word, dictionary.get('UNK'))
    word_row = embeddings_df.iloc[word_mapping].values
    embeddings_df = get_closest_words_to_vector(word_row, embeddings_df)
    embeddings_df = embeddings_df.sort_values(by='similarity').head(num_of_words)
    return embeddings_df

def get_closest_words_to_vector(input_vector, embeddings_df):
    embeddings_df['similarity'] = embeddings_df.apply(lambda row: distance.pdist(np.vstack((row, input_vector)))[0], axis = 1)
    return embeddings_df

def get_closest_words(input_word, embeddings_df, dictionary, reverse_dictionary, num_of_words = 10):
    df = get_closest_word_embeddings_df(input_word, embeddings_df, dictionary, num_of_words)
    results = []
    for i, v in df.iterrows():
        results.append((reverse_dictionary[i], v['similarity']))

def get_closest_word_in_chunk(input_word, embeddings_df, dictionary):
    word_mapping = dictionary.get(input_word, dictionary.get('UNK'))
    word_row = embeddings_df.iloc[word_mapping].values
    embeddings_df['similarity'] = embeddings_df.apply(lambda row: distance.pdist(np.vstack((row, word_row)))[0], axis = 1)
    embeddings_df = embeddings_df.sort_values(by='similarity').head(10)
    return embeddings_df

def test_sentence(input_str, dictionary, max_length = 50):
    input_tokens = nltk.tokenize.word_tokenize(input_str)
    input_tokens = input_tokens[-n_input:]
    word_ids = [dictionary.get(i, dictionary.get('UNK')) for i in input_tokens]

    with tf.Session() as sess:
        pass
        #saver.restore(sess, "models/lstm_model")

def generate_inputs(embeddings_df, data, dictionary, max_len = ):
    pass

def get_model(x, weights, biases):
    x = tf.reshape(x, [-1, full_input_size])
    x = tf.split(x, n_input, 1)
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def train_model(training_data, embeddings_df, count, data, dictionary, reverse_dictionary):
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([vocab_size]))
    }
    x = tf.placeholder("float", [None, full_input_size, 1])
    y = tf.placeholder("float", [None, vocab_size])
    model = get_model(x, weights, biases)

    correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=model, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver()
        step = 0
        offset = random.randint(0, n_input + 1)
        acc_total = 0
        loss_total = 0

        while step < training_iters:
            #random_start = random.randint(1, len(data)-n_input-1)
            #symbols_in_keys = [data[i] for i in range(random_start, random_start + n_input)]
            random_start = random.randint(1, len(training_data) - n_input - 1)
            symbols_in_keys = [training_data[i] for i in range(random_start, random_start + n_input)]
            in_np_array = np.array(symbols_in_keys)
            symbols_in = np.reshape(in_np_array, (-1, full_input_size, 1))

            #symbols_out = np.reshape(embeddings_df.loc[data[random_start + n_input]].values, (1, -1))
            symbols_out = np.reshape(np.array(training_data[random_start + n_input]), [1, -1])
            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, model], \
                                                    feed_dict={x: symbols_in, y: symbols_out})

            loss_total += loss
            acc_total += acc
            if step%display_step == 0:
                print(step, acc_total/display_step, loss_total/display_step)
                loss_total = 0
                acc_total = 0

            step += 1
        saver.save(session, 'models/lstm_model')

def main():
    count, data, dictionary, reverse_dictionary, final_embeddings = load_models()
    embeddings_df = pd.DataFrame(final_embeddings)
    np_training_data = get_training_data(data, dictionary, embeddings_df)
    train_model(np_training_data, embeddings_df, count, data, dictionary, reverse_dictionary)

    test_sentence('EA has great games and ethical practices.' + new_comment_marker, dictionary)


if __name__ == '__main__':
    main()
    #main()
    #a = get_closest_words('the', embeddings_df, dictionary, reverse_dictionary)
