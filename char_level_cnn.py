import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, Conv1D, Dense,Dropout, Activation, MaxPooling1D, Flatten
from tensorflow.python.framework.func_graph import convert_structure_to_signature


class Char_level_CNN(tf.keras.Model):
    def __init__(self,input_size,vocab_size,embed_size,embedding_weights,conv_layers,fully_connected_layers,dropout_p,**kwargs):
        """
        Args:
            input_size : number of characters used to predict the next character
            vocab_size : number of characters in the alphabet
            embed_size : number of dimentions of the output layer
            embedding_weights : numpy matrix with the initial embedding_weights to be used to
            conv_layers : list of parameters of convolution layers. 4 values should be specified
                * number of filters
                * filters' size
                * pooling size of a filter ???Have no idea whtat this is...
                * name of each filter
            fully_connected_layers : parameters of fully connected layers to be used
            dropout_p : p parameter of the droupout layer
        """
        super().__init__(self,**kwargs)
        self.input_size = input_size
        self.vocab_size = vocab_size 
        self.emebed_size = embed_size
        self.embedding_weights = embedding_weights
        self.conv_layers = conv_layers
        self.fully_connected_layers = fully_connected_layers
        self.dropout_p = dropout_p
        
        #self.inputs = inputs #Input(shape=(self.input_size,), name="input", dtype="int64")
        self.embedding = Embedding(self.vocab_size+1, self.emebed_size, input_length=self.input_size, weights = [self.embedding_weights], name="embedding")
        
        self.conv = []
        self.activation = []
        self.max_pooling = []

        for filter_num, filter_size, pooling_size,name in self.conv_layers:
            self.conv.append(Conv1D(filter_num,filter_size, name=name+"_convolution"))
            self.activation.append(Activation("relu",name=name+"_activation"))
            if pooling_size!=-1:
                self.max_pooling.append(MaxPooling1D(pool_size=pooling_size,name=name+"_max_pooling"))

        self.flatten = Flatten()

        self.dense = []
        self.dropout = []

        for dense_size in self.fully_connected_layers:
            self.dense.append(Dense(dense_size, activation="relu"))
            self.dropout.append(Dropout(self.dropout_p))

        self.predictions = Dense(self.vocab_size,activation="softmax")

    def call(self,inputs):
        x = self.embedding(inputs)
        conv_count = 0
        max_pool_count = 0
        for _filter_num, _filter_size, pooling_size,_name in self.conv_layers:
            x = self.conv[conv_count](x)
            x = self.activation[conv_count](x)
            if pooling_size!=-1:
                x = self.max_pooling[max_pool_count](x)
                max_pool_count += 1
            conv_count += 1

        x = self.flatten(x)

        for ds,dp in zip(self.dense,self.dropout):
            x = ds(x)
            x = dp(x)
        

        predictions = self.predictions(x)

        return predictions