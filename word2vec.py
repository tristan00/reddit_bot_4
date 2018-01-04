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

paragraph_marker = ' marker_new_paragraph '
new_comment_marker = ' marker_new_comment '
tab_marker = ' marker_new_paragraph '
return_marker = ' marker_return '
vocabulary_size = 100000
db_location = r'C:\Users\tdelforge\Documents\project_dbs\reddit\reddit.db'
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 64
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
data_index = 0


def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    print('building dict')
    for i, (word, _) in enumerate(count):
        dictionary[word] = len(dictionary)
        if i %1000 == 0:
            print('building dict', i, len(count), time.time())
    data = list()
    unk_count = 0
    print('mapping data')
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
        if i % 1000 == 0:
            print('mapping data', i, len(count), time.time())
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

        comments_df = pd.read_sql('select * from comments order by p_id', conn)
        comment_chains = []
        for count, (p_id, title) in enumerate(posts):
            if count > 50000:
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
        tokenized_inputs = []
        print('tokenizing')
        for count, i in enumerate(comment_chains):
            tokenized_inputs.append(get_input_text_from_comment_chain(i))
            if count %1000 == 0:
                print('tokenized:', count, time.time())
        # get_features_from_inputs()
        return tokenized_inputs

def generate_batch(batch_size, num_skips, skip_window, data):
    global data_index
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
            for word in data[:span]:
                buffer.append(word)
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

def run_model(reverse_dictionary,data):
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
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        init = tf.global_variables_initializer()

    num_steps = 1000001

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window, data)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 1000 == 0:
                if step > 0:
                    average_loss /= 1000
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
        return normalized_embeddings.eval()

def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
         xy=(x, y),
         xytext=(5, 2),
         textcoords='offset points',
         ha='right',
         va='bottom')
    plt.savefig(filename)

def save_vocad(data, count, dictionary, reverse_dictionary):
    with open('models/data.plk', 'wb') as f:
        pickle.dump(data, f)
    with open('models/count.plk', 'wb') as f:
        pickle.dump(count, f)
    with open('models/dictionary.plk', 'wb') as f:
        pickle.dump(dictionary, f)
    with open('models/reverse_dictionary.plk', 'wb') as f:
        pickle.dump(reverse_dictionary, f)

def get_saved_vocab():
    with open('models/data.plk', 'rb') as f:
        data = pickle.load(f)
    with open('models/count.plk', 'rb') as f:
        count = pickle.load(f)
    with open('models/dictionary.plk', 'rb') as f:
        dictionary = pickle.load(f)
    with open('models/reverse_dictionary.plk', 'rb') as f:
        reverse_dictionary = pickle.load(f)
    return data, count, dictionary, reverse_dictionary

def main():
    try:
        data, count, dictionary, reverse_dictionary = get_saved_vocab()
    except:
        traceback.print_exc()
        inputs = get_data()
        vocab = []
        for count, i in enumerate(inputs):
            vocab.extend(i)
            if count %1000 == 0:
                print('splitting sentences:', i, len(inputs))
        data, count, dictionary, reverse_dictionary = build_dataset(vocab, vocabulary_size)
        save_vocad(data, count, dictionary, reverse_dictionary)
        del vocab

    try:
        with open('models/final_embeddings.plk', 'rb') as f:
            final_embeddings = pickle.load(f)
    except:
        print('Most common words (+UNK)', count[:5])
        print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

        batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1, data=data)
        for i in range(8):
            print(batch[i], reverse_dictionary[batch[i]],
                  '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
        final_embeddings = run_model(reverse_dictionary=reverse_dictionary,data=data)

        with open('models/final_embeddings.plk', 'wb') as f:
            pickle.dump(final_embeddings, f)

    print('making graph')
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=10000)
    plot_only = 1000
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels,  'tsne.png')


if __name__ == '__main__':
    main()



