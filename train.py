import sys
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from model import getmodel
from parsechat import create_indexes_tape, create_datasets
from utils import create_lr_sched, export_vocabulary, export_embeddings


def check_dirs(run_hash):
    logs_path = os.path.join('logs/', run_hash)
    ckp_path = os.path.join('ckp/', run_hash)
    tok_path = os.path.join('ckp/', run_hash+'.tokenizer.json')
    try:
        os.listdir(logs_path)
    except FileNotFoundError:
        os.makedirs(logs_path)

    return logs_path, ckp_path, tok_path


def train(dataset_path,
        run_hash, 
        seq_len=32, 
        vocab_size=10000,
        emb_dim=32,
        batch_size=128,
        epochs=20,
        train_split = 0.8,
        val_split = 0.2):
    
    logs_path, ckp_path, tok_path = check_dirs(run_hash)

    ckp_cb = tf.keras.callbacks.ModelCheckpoint(
        ckp_path,
        'val_accuracy', 
        save_best_only=False,
        save_weights_only=True)

    lr_cb = tf.keras.callbacks.LearningRateScheduler(
        create_lr_sched(epochs/2, epochs), True)

    tb_cb = tf.keras.callbacks.TensorBoard(
            logs_path, 10, True, True, 
            embeddings_freq=10,  
            embeddings_metadata=logs_path+'/meta.tsv')

    with open(tok_path, 'r') as f:
        tokenizer = tokenizer_from_json(f.read())

    indexes_tape = create_indexes_tape(dataset_path, tokenizer)
    train_nbatches = int((len(indexes_tape)-seq_len) * train_split / batch_size)
    val_nbatches = int((len(indexes_tape)-seq_len) * val_split / batch_size)
    
    train_ds, val_ds = create_datasets(
        indexes_tape,
        train_nbatches, 
        val_nbatches,
        batch_size, 
        seq_len, 
        vocab_size)

    model = getmodel(seq_len-1, vocab_size, emb_dim, ckp_path)

    embeddings = model.layers[0].weights[0].numpy() 
    export_vocabulary(vocab_size, tokenizer.word_index, logs_path)
    export_embeddings(embeddings, logs_path)

    hist = model.fit(
        train_ds, 
        batch_size=batch_size, 
        epochs=epochs,
        steps_per_epoch=train_nbatches,
        validation_data=val_ds, 
        callbacks=[ckp_cb, lr_cb, tb_cb])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for training the NLP model')
    parser.add_argument('dataset', type=str)
    parser.add_argument('run_hash', type=str)
    parser.add_argument('epochs', type=int)
    args = parser.parse_args()

    train(args.dataset, args.run_hash, epochs=args.epochs)

