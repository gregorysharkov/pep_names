import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

from tensorflow.python.framework.func_graph import convert_structure_to_signature

class Char_level_bidirectional(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units

        self.embedding = Embedding(input_dim=self.vocab_size,output_dim=self.embedding_dim)
        self.lstmb = Bidirectional(
            LSTM(self.rnn_units,
            return_sequences=False,
            return_state=True,
            dropout=0.1)
        )
        self.dense = Dense(vocab_size)


    def call(self, inputs, hidden=None):
        x = inputs
        x = self.embedding(x)
        
        if hidden is None:
            hidden = self.initialize_hidden_state()
               
        output, forward_h, forward_c, backward_h, backward_c = self.lstmb(x, initial_state=hidden)
        return output, forward_h, forward_c, backward_h, backward_c

    def initialize_hidden_state(self):
        init_state =  tf.zeros((10,10))
        #[tf.zeros((self.rnn_units, self.rnn_units)) for i in range(2)]
        #[tf.zeros((self.rnn_units, self.rnn_units)) for i in range(2)]
        return init_state        
        
        # if states is None:
        #     states = self.bidirectional.get_initial_state(x)

        # x, states = self.bidirectional(x, initial_state=states,training=training)
        # x = self.dense(x)
        # if return_state:
        #     return x, states
        # else:
        #     return x