import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
import nltk
import re

learning_rate = 0.001
training_iters = 100000
display_step = 1000
n_input = 8
n_hidden = 1024
max_vocab_size = 50000
pos_size = 36
total_vocab_size =  max_vocab_size + pos_size

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
    model  = get_model(x, weights, biases)