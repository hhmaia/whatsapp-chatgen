import os
import sys
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from model import getmodel
from parsechat import parse_tape, create_indexes_tape


def generate_sentences(model_path, seed_path, num_words):
    model = getmodel(31, 10000, 32, model_path)
    tokenizer_path = model_path + '.tokenizer.json' 

    with open(tokenizer_path, 'r') as f:
        tokenizer = tokenizer_from_json(f.readlines()[0])

#    with open(seed_path, 'r') as f:
#        seed = f.readlines()[0]

    seed = create_indexes_tape('seed3', tokenizer)
    seed_seq = seed
#    seed_seq = tokenizer.texts_to_sequences([seed])[0]
#    seed_seq = tf.keras.preprocessing.sequence.pad_sequences([seed_seq], 31)[0]
    seed_seq = list(seed_seq)
    pred = None 
    out_seq = []

    for _ in range(num_words):
        seed_seq.extend(out_seq)
        seq_input = seed_seq[-(31):]
        seq_input = np.expand_dims(seq_input, 0)
        res = model.predict([seq_input], 1)
        pred = res.squeeze().argmax()
        out_seq.append(pred)

    words = [tokenizer.index_word[w] for w in out_seq
            if w not in [0]]
    sentence = ' '.join(words).replace('<eom>', '\n')
    print(sentence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat sentences generator')
    parser.add_argument('model_path', type=str)
    parser.add_argument('seed_path', type=str)
    parser.add_argument('num_words', type=int)
    args = parser.parse_args()
    generate_sentences(args.model_path, args.seed_path, args.num_words)

