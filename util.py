import string

import numpy as np


def init_weights(fi, fo):
    return np.random.randn(fi, fo) / np.sqrt(fi + fo)


def remove_punctuation(s):
    return s.translate(string.punctuation)


def get_robert_frost_word_indexes():
    word2idx = {'START': 0, 'END': 1}
    current_idx = 2
    sentences = []
    for line in open('robert_frost_poem.txt'):
        line = line.strip()
        if line:
            tokens = remove_punctuation(line.lower()).split()
            sentence = []
            for t in tokens:
                if t not in word2idx:
                    word2idx[t] = current_idx
                    current_idx += 1
                idx = word2idx[t]
                sentence.append(idx)
            sentences.append(sentence)
    return sentences, word2idx


