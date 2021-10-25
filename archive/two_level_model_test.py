import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import initializers, regularizers, constraints
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
tf.config.run_functions_eagerly(True)
from typing import Iterable
from dataclasses import dataclass

from tensorflow.tools.docs.doc_controls import doc_in_current_and_subclasses

@dataclass
class BiGruSettings:
    name : str
    input_shape: tuple[int]
    n_units: int

@dataclass 
class DistanceSettings:
    name : str
    mode: str

@dataclass
class InnerModelSettings:
    name: str
    embedding_input_dim : int
    embedding_output_dim : int
    char_level_settings: BiGruSettings
    word_level_settings: BiGruSettings

@dataclass
class OuterModelSettings:
    name: str
    inner_settings: InnerModelSettings
    distance_settings: DistanceSettings
    n_dense_units: int

class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, return_attention=False,
                 **kwargs):
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name=f'{self.name}_Weight',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name=f'{self.name}_bias',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]

class BiGruWithAttention(Layer):
    def __init__(self,settings:BiGruSettings,**kwargs):
        super().__init__(**kwargs)
        self.settings = settings
        self.gru_layer = tf.keras.layers.GRU(self.settings.n_units, return_sequences=True, name=self.settings.name+"_gru")
        self.bidirectional_layer = tf.keras.layers.Bidirectional(self.gru_layer, input_shape=self.settings.input_shape, name=self.settings.name+"_BiGru")
        self.attention_layer = Attention(return_attention=True, name = self.settings.name+"_attention_vec")
    
    def call(self,x,training=False):
        if len(x.shape) == 2:
            x = tf.expand_dims(x,axis=0)
        print(x.numpy)
        x = self.bidirectional_layer(x, training=training)
        sequence, element_scores = self.attention_layer(x,training=training)

        return sequence

class DistanceLayer(tf.keras.layers.Layer):
    """
    Layer responsible for computation of cosine similarity
    """
    def __init__(self,settings:DistanceSettings,**kwargs):
        super().__init__(**kwargs)
        self.settings = settings

    def call(self,inputs):
        input_a = inputs[0]
        input_b = inputs[1]

        if self.settings.mode == "abs":
            dist = tf.math.abs(tf.keras.slosses.cosine_similarity(input_a,input_b))
        elif self.settings.mode == "zero_to_one":
            dist = ( 1-tf.keras.losses.cosine_similarity(input_a,input_b) ) / 2
        else:
            dist = tf.keras.losses.cosine_similarity(input_a,input_b)

        return (dist)

class InnerModel(Layer):
    def __init__(self,settings:InnerModelSettings,**kwargs):
        super().__init__(**kwargs)

        self.settings = settings
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=self.settings.embedding_input_dim,output_dim=self.settings.embedding_output_dim)
        self.char_level_attention = BiGruWithAttention(self.settings.char_level_settings)
        self.word_level_attention = BiGruWithAttention(self.settings.word_level_settings)

    def call(self, x, training=False,debug=False):
        if debug:
            print(f"Input: {x}")
        x = self.embedding_layer(x,training=training)
        if debug:
            print(f"X after embedding: {x}")
        x = self.char_level_attention(x)
        if debug:
            print(f"X after char level attention: {x}")
        x = self.word_level_attention(x)
        if debug:
            print(f"X after word level attention: {x}")

        return x

class OuterModel(Model):
    """
    Outer model that is reponsible for processing of tuples of string couples
    """
    def __init__(self,settings:OuterModelSettings,**kwargs):
        super().__init__(**kwargs)
        self.settings = settings
        self.inner_model = InnerModel(self.settings.inner_settings)
        self.distance_layer = DistanceLayer(self.settings.distance_settings)
        self.dense_layer = tf.keras.layers.Dense(self.settings.n_dense_units)

    def call(self, inputs, training=False, debug=True):
        input_a = inputs[0]
        input_b = inputs[1]
        if debug:
            print(f"{input_a=}")
            print(f"{input_b=}")


        repr_a = self.inner_model(input_a, training=training, debug=debug)
        repr_b = self.inner_model(input_b, training=training, debug=debug)
        if debug:
            print(f"{repr_a=}")
            print(f"{repr_b=}")

        distance = self.distance_layer((repr_a,repr_b))
        if debug:
            print(f"{distance=}")

        output = self.dense_layer(tf.expand_dims(distance,axis=1))

        return output

def preprocess_list(lst:Iterable, tokenizer:Tokenizer, max_words:int=None, max_char:int=None):
    """
    Function preprocesses a given list. A string is turned into a sequence of words of length max_words,
    then each word is turned into a sequence of characters of length max_char, if any of maximum parameters is 
    not specified, they are calcualted based on the provided list

    function returns a tf.DataSet
    """
    if max_words is None:
        word_counts = [len(x.split()) for x in lst]
        max_words = max(word_counts)

    if max_char is None:
        words = list(set([word for name in lst for word in name.split()]))
        word_lengths = [len(word) for word in words]
        max_char = max(word_lengths)

    padded_test_string = [x + " "*(max_words-len(x.split())) for x in lst]
    test_split = [x.split(" ") for x in padded_test_string]
    test_sequences = [tokenizer.texts_to_sequences(x) for x in test_split]
    padded_test_sequences = [pad_sequences(x,maxlen=max_char,padding="post") for x in test_sequences]
    return_matrix = tf.data.Dataset.from_tensor_slices(padded_test_sequences)
    return return_matrix


def main():
    test_string = ["Grigory Sharkov","Boris Jonson Junior"]
    test_combination = ["Grigory Sharkov","Moris Donson Junior"]
    test_match = [1,0]

    tokenizer = Tokenizer(char_level=True,lower=False)
    tokenizer.fit_on_texts(test_string+test_combination)
    print(tokenizer.word_index)

    test_names = preprocess_list(test_string,tokenizer,3,4)
    test_combinations = preprocess_list(test_combination,tokenizer,3,4)
    features = tf.data.Dataset.zip((test_names,test_combinations))
    match = tf.data.Dataset.from_tensor_slices(np.array(test_match))
    training_data = tf.data.Dataset.zip((features,match)).batch(1)

    experiment_settings = OuterModelSettings(
        name="two_level_rnn_with_attention",
        inner_settings=InnerModelSettings(
            name="inner_model",
            embedding_input_dim=len(tokenizer.word_index),
            embedding_output_dim=5,
            char_level_settings=BiGruSettings(
                name="char_bigru",
                input_shape=(3,4,5),
                n_units=6
            ),
            word_level_settings=BiGruSettings(
                name="word_bigru",
                input_shape=(3,12),
                n_units=7
            ),
        ),
        distance_settings = DistanceSettings("distance","zero_to_one"),
        n_dense_units=1
    )

    model = OuterModel(experiment_settings)
    model.compile(optimizer="Nadam",loss="binary_crossentropy")
    print(model.predict(features))
    # for el in features:
    #     print(model.predict(el))
    # model.fit(training_data)


if __name__ == '__main__':
    main()