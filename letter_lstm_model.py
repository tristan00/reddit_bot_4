
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
alphabet_size = 26

def get_letter_embedding(input_letter):
    lower_case_letter = input_letter.lower()
    letter_index = ord(lower_case_letter)- ord('a')

    if letter_index > alphabet_size-1 or letter_index < 0:
        letter_index = alphabet_size
    if input_letter.isupper():
        upper_case = 1
    else:
        upper_case = 0
    output_list = [0 for _ in range(alphabet_size + 1)]
    output_list[letter_index] = 1
    output_list + [upper_case]

    return np.array([])


def get_letter_embeddings(chain):
    comments = chain.split(new_comment_marker)
    embeddings_output = []

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