import io
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_hist(history, key, path, with_val=True, sufix=''):
    train_series = history.history[key]
    epochs = range(len(train_series))
    plt.xlabel('Epochs')
    plt.ylabel(key)
    plt.plot(epochs, train_series, color='blue')

    if with_val:
        val_series = history.history['val_' + key]
        plt.plot(epochs, val_series, color='red')
        plt.legend(['training', 'validation'])
    else:
        plt.legend(['training'])

    return plt.show()


def export_vocabulary(vocab_size, word_index, dir_path):
    with open(os.path.join(dir_path, 'meta.tsv'), 'w') as f:
        f.writelines(['0\n'])
        words = list(word_index.keys())
        f.write('\n'.join(words[:vocab_size]))

                
def export_embeddings(embeddings, dir_path):
    with open(os.path.join(dir_path, 'vect.tsv'), 'w') as f: 
        text = '\n'.join(
               '\t'.join(str(v) for v in e)
               for e in embeddings)
        f.write(text)
     

def plot_series(x, y, scale='log'):
    fig = plt.figure()
    sub = fig.add_subplot()
    sub.set_yscale(scale)
    sub.plot(x, y)
    plt.show()


def create_lr_sched(start_epoch, n_epochs, lr0=1e-3, lr_end=1e-9):
    """
    start_epoch: epoch where to start decaying
    n_epochs: total number of epochs
    lr0: initial learning rate
    lr_end: learning rate end value

    return: learning rate scheduler function with given parameters.
    """

    def sched(epoch):
        exp_range = np.log10(lr0/lr_end) 
        epoch_ratio = (epoch - start_epoch)/(n_epochs - start_epoch)
        if epoch < start_epoch:
            lr = lr0
        else:
            lr = lr0 * 10**-(exp_range * epoch_ratio) 
        return lr

    return sched


