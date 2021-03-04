import argparse
import tensorflow as tf
import tensorflow.keras.layers as klayers


def export_model(path, input_seq_len, vocab_size, emb_dim):
   
    model = tf.keras.Sequential([
        klayers.Embedding(vocab_size+1, emb_dim, input_length=input_seq_len),
        klayers.Bidirectional(
            klayers.LSTM(32, return_sequences=True)),
        klayers.BatchNormalization(),
        klayers.Bidirectional(
            klayers.LSTM(32, return_sequences=True)),
        klayers.BatchNormalization(),
        
        klayers.Bidirectional(klayers.LSTM(32)),
        klayers.BatchNormalization(),
        klayers.Flatten(),
        klayers.Dense(64, 'relu'),
        klayers.Dense(vocab_size, 'softmax')
    ])

    model.summary()
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
    model.save(path)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Create a keras model and save it in tf format.')
    parser.add_argument('path', type=str)
    parser.add_argument('seq_len', type=int)
    parser.add_argument('vocab_size', type=int)
    parser.add_argument('emb_dim', type=int)
    args = parser.parse_args()
    
    export_model(args.path, args.seq_len, args.vocab_size, args.emb_dim)

