import tensorflow as tf
import tensorflow.keras.layers as klayers


def getmodel(input_seq_len, vocab_size, emb_dim, ckp_path):
   
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

    try:
        model.load_weights(ckp_path)
        print('Checkpoint loaded')
    except:
        print('Error loading checkpoint.')

    model.summary()
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
    return model
