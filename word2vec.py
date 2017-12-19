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
import math

paragraph_marker = ' marker_new_paragraph '
new_comment_marker = ' marker_new_comment '
tab_marker = ' marker_new_paragraph '
return_marker = ' marker_return '
vocabulary_size = 50000
db_location = r'C:\Users\tdelforge\Documents\project_dbs\reddit\reddit.db'
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 64
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

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
    #get_features_from_inputs()
    return tokenized_inputs

def generate_batch(batch_size, num_skips, skip_window, data, data_index):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

def run_model(reverse_dictionary,data, data_index):
    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.device('/cpu:0'):
            embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
            stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        init = tf.global_variables_initializer()

    num_steps = 100001

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window, data, data_index)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()


def main():
    inputs = get_data()
    vocab = functools.reduce(operator.concat, inputs)
    data, count, dictionary, reverse_dictionary = build_dataset(vocab, vocabulary_size)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1, data=data, data_index=data_index)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]],
              '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
    run_model(reverse_dictionary=reverse_dictionary,data=data, data_index=data_index)


if __name__ == '__main__':
    main()



