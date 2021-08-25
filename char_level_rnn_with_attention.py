import tensorflow as tf
import pandas as pd
import numpy as np
from inner_model_settings import InnerModelSettings

#custom layers
class Attention(tf.keras.Model):
    """
    Attention layer for inner model
    """
    def __init__(self,units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)


    def call(self, features, hidden, training=False):
        # hidden shape == (batch_size, hidden size)
        # hidden with time axis shape == (batch_size, 1, hidden_size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden,1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = tf.nn.tanh(
            self.W1(features,training=training) + self.W2(hidden_with_time_axis,training=training)
        )
        # attention weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector#, attention_weights


class DistanceLayer(tf.keras.layers.Layer):
    """
    Layer responsible for computation of cosine similarity
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def call(self,input_a,input_b):
        dist = ( 1-tf.keras.losses.cosine_similarity(input_a,input_b) ) / 2
        return dist #tf.reshape(dist,shape=(len(dist),-1))


# inner model with attention
class InnerModel(tf.keras.Model):
    """
    Inner model to be used inside the outer model.
    It is responsible for transformations of a sequence into a vector representation
    that will be used further for comparisson
    """
    def __init__(self, settings: InnerModelSettings):
        super().__init__()
        self.n_embedding_dims = settings.n_embedding_dims
        self.n_gru = settings.n_gru
        self.n_dense = settings.n_dense
        self.n_units_attention = settings.n_units_attention
        self.input_embedding = settings.input_embedding

        self.embedding = tf.keras.layers.Embedding(self.input_embedding+1,self.n_embedding_dims,name="inner_embedding")
        self.dense_embedding = tf.keras.layers.Dense(self.n_embedding_dims*2, name="inner_dense_after_embedding")
        self.dropout = tf.keras.layers.Dropout(.5)
        self.gru_1 = tf.keras.layers.GRU(self.n_gru,name="inner_gru_1",return_sequences=True)
        self.gru_2 = tf.keras.layers.GRU(self.n_gru,name="inner_gru_2",return_sequences=True, return_state=True)
        self.attention = Attention(self.n_units_attention)
        self.bi_gru_1 = tf.keras.layers.Bidirectional(self.gru_1,name="inner_bidirectional_1")
        self.bi_gru_2 = tf.keras.layers.Bidirectional(self.gru_2,name="inner_bidirectional_2")
        self.normalizer = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(self.n_dense,name="inner_dense")
    
    def call(self, x, training=False):
        embedded_strings = self.embedding(x,training=training)
        embedded_strings = self.dense_embedding(embedded_strings, training=training)
        embedded_strings = self.dropout(embedded_strings, training=training)
        after_bigru = self.bi_gru_1(embedded_strings,training=training)
        (bigru_output, state_h, state_c) = self.bi_gru_2(after_bigru, training=training)
        context_vector = self.attention(bigru_output,state_h)
        context_normalized = self.normalizer(context_vector,training=training)
        final_embedding = self.dense(context_normalized,training=training)

        return self.dense(final_embedding)


#outer model to be used for training
class OuterModel(tf.keras.Model):
    """
    Outer model. It takes two inputs (one sequence for each string compared)
    trahsorms them into a vector representations using inner model
    and comptes the output using the cosine distance layer
    """
    def __init__(self, settings: InnerModelSettings):
        super().__init__()
        print(settings.input_embedding)

        #layers
        self.inner_model = InnerModel(settings)
        self.distance_layer = DistanceLayer()
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid",name="output_layer")

        #learned representations
        self.repr_a = []
        self.repr_b = []
        self.cosine_similarity = 0


    def call(self,inputs,training=False):
        input_a = inputs[0]
        input_b = inputs[1]

        self.repr_a = self.inner_model(input_a, training=training)
        self.repr_b = self.inner_model(input_b, training=training)
        self.cosine_similarity = self.distance_layer(self.repr_a,self.repr_b,training=training)
        #output = self.output_layer(tf.expand_dims(self.cosine_similarity,1))
        return tf.expand_dims(self.cosine_similarity,1) #output
        # self.output_layer(tf.reshape(self.cosine_similarity,shape=(1,len(self.cosine_similarity))), training=training) #self.output_layer(self.cosine_similarity,training=training)


    def train_step(self, data):
        x,y = data

        with tf.GradientTape() as tape:
            #predict the value
            y_pred = self(x, training=True)
            #compute the loss
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        #compute gradients
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        #update weights
        self.optimizer.apply_gradients(zip(gradients,trainable_variables))
        #update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}



