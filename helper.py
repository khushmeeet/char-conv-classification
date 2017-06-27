import numpy as np
import string
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

char_limit = 1014


def get_data(path):
    labels = []
    inputs = []
    df = pd.read_csv(path, names=['one','second','third'])
    df = df.drop('second', axis=1)
    data = df.values
    for label,text in data:
        inputs.append(text.lower())
        if label == 1:
            labels.append([1, 0, 0, 0])
        elif label == 2:
            labels.append([0, 1, 0, 0])
        elif label == 3:
            labels.append([0, 0, 1, 0])
        else:
            labels.append([0, 0, 0, 1])
    return inputs, np.array(labels, dtype=np.float32)


def create_vocab_set(inputs):
    vocab = set()
    for i in inputs:
        for j in i.split(' '):
            vocab.add(j)
    vocab_size = len(vocab)
    word2idx = {}
    for i, c in enumerate(vocab):
        word2idx[c] = i
    return vocab, vocab_size, word2idx


def _encode_text(s, word2idx):
    vec = []
    for i in s.split(' '):
        vec.append(word2idx[i])
    return np.array(vec)


def get_encoded_text(text, word2idx, sent_limit):
    encoded_text = []
    for single_text in text:
        encoded_text.append(_encode_text(single_text, word2idx))
    encoded_text = pad_sequences(encoded_text, maxlen=sent_limit, value=0.)
    return np.array(encoded_text)


def batch_gen(encoded_text, labels, batch_size):
    for ii in range(0, len(encoded_text), batch_size):
        x = encoded_text[ii:ii + batch_size]
        y = labels[ii:ii + batch_size]
        yield (x, y)