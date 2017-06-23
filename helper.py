import numpy as np
import string
import os

PATH = 'training_data_5151/'
#
# vocab = list(string.ascii_lowercase) + list(string.punctuation) + list(string.digits) + ['\n']
#
# vocab_size = len(vocab)
char_limit = 1014


def get_data():
    text = []
    labels = []
    dir_list = os.listdir(PATH)
    for i in dir_list:
        if i == 'neg':
            files = os.listdir(PATH+i+'/')
            for j in files:
                with open(PATH+i+'/'+j, encoding='utf-8') as f:
                    text.append(f.read())
                    labels.append([1,0,0])
        if i == 'neu':
            files = os.listdir(PATH+i+'/')
            for j in files:
                with open(PATH+i+'/'+j, encoding='utf-8') as f:
                    text.append(f.read())
                    labels.append([0,1,0])
        if i == 'pos':
            files = os.listdir(PATH+i+'/')
            for j in files:
                with open(PATH+i+'/'+j, encoding='utf-8') as f:
                    text.append(f.read())
                    labels.append([0,0,1])
    return text, labels


text, labels = get_data()


def create_vocab_set(text):
    char2idx = {}
    reverse_char2idx = {}
    vocab = set()
    for c in text:
        for i in c:
            vocab.add(i)
    vocab_size = len(vocab)
    for i, c in enumerate(vocab):
        char2idx[c] = i
        reverse_char2idx[i] = c

    return vocab, vocab_size, char2idx, reverse_char2idx


def _encode_char(s, vocab_size, char2idx):
    char_vec = []
    for c in s[:char_limit]:
        vec = np.zeros(vocab_size)
        vec[char2idx[c]] = 1
        char_vec.append(vec)
    return np.array(char_vec)


def get_encoded_text(text, vocab_size, char2idx):
    encoded_text = []
    for single_text in text:
        encoded_text.append(_encode_char(single_text, vocab_size, char2idx))
    return encoded_text


def batch_gen(encoded_text, labels, batch_size):
    for ii in range(0, len(encoded_text), batch_size):
        x = encoded_text[ii:ii + batch_size]
        y = labels[ii:ii + batch_size]
        yield (x, y)