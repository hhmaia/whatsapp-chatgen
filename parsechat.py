import os
import sys
import argparse
import re
from pprint import pprint as pp

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

from regex_token import _regex_token


def _replace_tokens(s, _regex_token): 
    x = s.lower()
    for p, t in _regex_token:
        x = re.sub(p, t, ' ' + x + ' ') 
    return x

def _msgs_gen(raw_lines):
    '''
    Strips date and time from the start of each message, and concatenates 
    newlines into a single message.

    '''
    date_exp = r'([0-3]?[\d]/){2}[\d][\d], [0-2][\d]:[\d]{2} - '
    pattern = re.compile(date_exp)
    
    msg = ''
    for line in raw_lines:
        match = pattern.match(line)
        if match:
            if msg != '':
                yield msg
            msg = line[match.end():]
        else:
            msg = ''.join([msg, line])


def _user_and_text_gen(msgs, max_text_len=128):
    """
    msgs: sentence with pattern 'r[\w ]+: ' to be parsed to user and text
    max_text_len: maximum length after pos trimming of msg text

    returns: (user, text)
    """
    msg_exp = r'[\w ]+: '
    pattern = re.compile(msg_exp)

    for msg in msgs:
        match = pattern.match(msg)
        if match:
            user = "".join(['<', match[0], '>'])
            user = user.replace(': ', '').replace(' ', '_')
            text = msg[match.end():]
        yield (user, text[:max_text_len])


def _parse_file(raw_lines):
    """
    returns: an iter for parsed and filtered messages.
    """
    msgs_iter = _msgs_gen(raw_lines)
    user_text_iter = _user_and_text_gen(msgs_iter)
    # user, text
    for u, t in user_text_iter: 
        text = _replace_tokens(t, _regex_token)
        if text != ' <MEDIA> ':
            yield ' '.join([u, text])


def parse_tape(dataset_path):
    with open(dataset_path, 'r') as f:
        next(f)
        parsed_file = _parse_file(f)
        msgs_tape = ' <EOM> '.join(parsed_file) 
    return msgs_tape


def export_tokenizer(filepath, dataset_path, max_vocab_size):
    msgs_tape = parse_tape(dataset_path)
    tokenizer = Tokenizer(max_vocab_size, filters='')
    tokenizer.fit_on_texts([msgs_tape])
    with open(filepath, 'w') as f:
        f.write(tokenizer.to_json())

    return tokenizer


def export_vocabulary(path, vocab_size, word_index):
    with open('path', 'w') as f:
        f.writelines(['0\n'])
        words = list(word_index.keys())
        f.write('\n'.join(words[:vocab_size]))


def create_indexes_tape(dataset_path, tokenizer):
    msgs_tape = parse_tape(dataset_path)
    return tokenizer.texts_to_sequences([msgs_tape])[0]


def create_datasets(tape,
                    train_n_batches,
                    val_n_batches,
                    batch_size,
                    seq_len):

    dataset = tf.data.Dataset.from_tensor_slices(tape)
    dataset = dataset.window(seq_len, 1, drop_remainder=True)
    dataset = dataset.flat_map(lambda x: x.batch(seq_len))
    dataset = dataset.map(lambda x: (x[:-1], x[-1]))
    dataset = dataset.shuffle((len(tape)-seq_len)//2, seed=1)
    dataset = dataset.batch(batch_size)
    train_dataset = dataset.take(train_n_batches)
    dataset = dataset.skip(train_n_batches)
    val_dataset = dataset.take(val_n_batches)
    train_dataset = train_dataset.repeat()
    
    return train_dataset, val_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Create tokenizer and save it as json.')
    parser.add_argument('tok_path', type=str)
    parser.add_argument('labels_path', type=str) 
    parser.add_argument('dataset', type=str)
    parser.add_argument('vocab_size', type=int)
    args = parser.parse_args()
    
    tokenizer = export_tokenizer(
            args.tok_path,
            args.dataset,
            args.vocab_size)

    export_vocabulary(
            args.labels_path,
            args.vocab_size,
            tokenizer.word_index) 

