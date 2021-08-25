import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GRU

class Char_level_rnn(tf.keras.Model):

    def __init__(self, vocab_size,embedding_dim,rnn_units):
        super().__init__(self)
        self.embedding = Embedding(vocab_size,embedding_dim)
        self.gru = GRU(rnn_units,return_sequences=False,return_state=True)
        self.dense = Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x)
        if states is None:
            states = self.gru.get_initial_state(x)
        
        x,states = self.gru(x, initial_state=states,training=training)
        x = self.dense(x)

        if return_state:
            return x, states
        else:
            return x

