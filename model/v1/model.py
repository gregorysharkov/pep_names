import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from model.v1.attention import Attention
from model.v1.model_settings import BiGruSettings, InnerModelSettings, DistanceSettings, OuterModelSettings

class BiGruWithAttention(Layer):
    """Attention layer. Runs a bidirectional GRU model, and applies attention to get one representation"""
    def __init__(self,settings:BiGruSettings,**kwargs):
        super().__init__(**kwargs)
        self.settings = settings
        self.gru_layer = tf.keras.layers.GRU(self.settings.n_units, return_sequences=True,input_shape=self.settings.input_shape, name=self.settings.name+"_gru")
        self.bidirectional_layer = tf.keras.layers.Bidirectional(self.gru_layer, name=self.settings.name+"_BiGru")
        self.attention_layer = Attention(return_attention=True, name = self.settings.name+"_attention_vec")
    
    def call(self,x,training=False):
        x = self.bidirectional_layer(x, training=training)
        sequence, element_scores = self.attention_layer(x,training=training)
        return sequence

    def compute_output_shape(self, input_shape):
        #2 types of transofmrations:
        #turn (None,5,5,10) into (None,5,10) and
        #turn (None,5,20) into (None,40)
        return (*input_shape[:-2],self.settings.n_units*2)

class InnerModel(Layer):
    """Inner layer responsible for taking one input, passing it through 2 bigru_attention layers and generating the representation"""
    def __init__(self,settings:InnerModelSettings,**kwargs):
        super().__init__(**kwargs)

        self.settings = settings

        self.embedding_layer = tf.keras.layers.Embedding(input_dim=self.settings.embedding_input_dim,output_dim=self.settings.embedding_output_dim)
        self.char_level_bigru = BiGruWithAttention(self.settings.char_level_settings,name="char_level_bigru")
        self.char_to_words_layer = tf.keras.layers.TimeDistributed(self.char_level_bigru,name="time_distributed_char_level")
        self.words_to_string_layer = BiGruWithAttention(self.settings.word_level_settings, name="word_level_bigru")

    def call(self, x, training=False,debug=False):
        if debug:
            print(f"input x: {x}")

        embedded_x = self.embedding_layer(x)
        if debug:
            print(f"{embedded_x=}")

        word_level = self.char_to_words_layer(embedded_x)
        if debug:
            print(f"{word_level=}")

        string_level = self.words_to_string_layer(word_level,training=training)
        if debug:
            print(f"{string_level=}")

        return string_level
    
    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return Model(inputs=[x], outputs=self.call(x))

class DistanceLayer(Layer):
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
            dist = tf.math.abs(tf.keras.losses.cosine_similarity(input_a,input_b,axis=1))
        elif self.settings.mode == "zero_to_one":
            dist = ( 1-tf.keras.losses.cosine_similarity(input_a,input_b,axis=1) ) / 2
        else:
            dist = tf.keras.losses.cosine_similarity(input_a,input_b,axis=1)

        #we want to limit values of this layer to be explicitly between 0 and 1
        dist = tf.expand_dims(dist,axis=1)
        epsilon = 0.0001
        dist = tf.clip_by_value(dist,epsilon,1-epsilon)

        return dist

class OuterModel(Model):
    """
    Outer model that is reponsible for processing of tuples of string couples
    """
    def __init__(self,settings:OuterModelSettings,**kwargs):
        super().__init__(**kwargs)
        self.settings = settings
        self.input_a = tf.keras.Input(shape=(settings.inner_settings.n_words,settings.inner_settings.n_characters),name='Input_a')
        self.input_b = tf.keras.Input(shape=(settings.inner_settings.n_words,settings.inner_settings.n_characters),name='Input_b')
        self.inner_model = InnerModel(self.settings.inner_settings)
        self.string_features = tf.keras.layers.Dense(160,activation="relu")
        self.distance_layer = DistanceLayer(self.settings.distance_settings)

    def call(self, inputs, training=False, debug=False):
        self.input_a = inputs[0]
        self.input_b = inputs[1]
        if debug:
            print(f"{self.input_a=}")
            print(f"{self.input_b=}")

        repr_a = self.inner_model(self.input_a, training=training, debug=debug)
        repr_a = self.string_features(repr_a,training=training)
        repr_b = self.inner_model(self.input_b, training=training, debug=debug)
        repr_b = self.string_features(repr_b,training=training)
        if debug:
            print(f"{repr_a=}")
            print(f"{repr_b=}")

        distance = self.distance_layer((repr_a,repr_b))
        if debug:
            print(f"{distance=}")

        return tf.squeeze(distance,axis=1)

    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return Model(inputs=[x], outputs=self.call(x))