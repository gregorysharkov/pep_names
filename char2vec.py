import pandas as pd
import numpy as np
import random as rd
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from char_level_cnn import Char_level_cnn
from char_level_rnn import Char_level_rnn
from char_level_bidirectional_rnn import Char_level_bidirectional

path = "data\\combinations\\"
true_data = pd.read_csv(path+"governors_true_match.csv",sep=";")
false_data = pd.read_csv(path+"governors_false_match.csv",sep=";")
combined_data = pd.concat([true_data,false_data])
names = sorted(set(list(combined_data.governor) + list(combined_data.combinations)))
words = sorted(set(word for name in list(map(str.split,names)) for word in name))
vocab = sorted(set(character for word in words for character in word))

train_words = rd.sample(words, round(len(words)*.6))
test_words = [x for x in words if x not in train_words]

tk = Tokenizer(num_words=None, char_level=True, oov_token="UNK")
tk.fit_on_texts(words)

train_sequences = tk.texts_to_sequences(train_words)
test_sequences = tk.texts_to_sequences(test_words)

MAX_LEN = 30
raw_train_data = np.array(pad_sequences(train_sequences, maxlen=MAX_LEN,padding="pre"),dtype="float32")
raw_test_data = np.array(pad_sequences(test_sequences, maxlen=MAX_LEN,padding="pre"),dtype="float32")

# how many characters use in the sequence
NUM_CHARS_TO_PREDICT = 10


def generate_training_data(sequences,chars_to_predict):
    """
    Function takes padded sequences and extracts equal length sequences as a feature vector and
    value of the next symbol as the predicted variable. it returns a 2 dimentional list:
    1. feature vector
    2. predicted variable

    length of sequences is defined by chars_to_predict
    """
    return_data = []
    for element in sequences:
        for i in range(len(element)):
            #if the element we are looking for is a padding symbol, or the remainig
            #number of symbols is less that we actually need, we stop
            if (element[-i-1] == 0.) or (i+chars_to_predict >= len(element)):
                break
            #otherwise we extract the feature
            return_data.append((element[-i-chars_to_predict-1:-i-1], element[-i-1]))

    return return_data

#Prepare training and test data
train_data = generate_training_data(raw_train_data, NUM_CHARS_TO_PREDICT)
test_data = generate_training_data(raw_test_data, NUM_CHARS_TO_PREDICT)

train_features = np.array(list(x[0] for x in train_data))
test_features = np.array(list(x[0] for x in test_data))

train_classes = to_categorical([int(x[1]-1) for x in train_data], num_classes=len(vocab)+1)
test_classes = to_categorical([int(x[1]-1) for x in test_data],num_classes=len(vocab)+1)

#wordking with embeddings
embedding_weights = []
embedding_weights.append(np.zeros(len(tk.word_index)))

for char,i in tk.word_index.items():
    onehot = np.zeros(len(tk.word_index))
    onehot[i-1] = 1
    embedding_weights.append(onehot)

embedding_weights = np.array(embedding_weights)

#now it is time to work on the CNN
input_size = NUM_CHARS_TO_PREDICT
vocab_size = len(tk.word_index)
embed_size = 128
conv_layers = [[64, 1, 3,"conv1"],
               [64, 1, 3,"conv2"],
               [64, 1, -1,"conv3"],
               [64, 1, -1,"conv4"],
               [64, 1, -1,"conv5"],
               [32, 1, 1,"conv6"]]

fully_connected_layers = [128, 128]
num_of_classes = len(vocab)+1
dropout_p = .5
optimizer = "adam"
loss = "categorical_crossentropy"


model = Char_level_bidirectional(
    vocab_size,
    embed_size,
    10
)

#inputs = Input(shape=(input_size,), name="input", dtype="int64")

# model = Char_level_cnn(
#     input_size,
#     vocab_size,
#     embed_size,
#     embedding_weights,
#     conv_layers,
#     fully_connected_layers,
#     dropout_p,
# )

#model = Char_level_rnn(vocab_size,256,input_size)
model.compile(optimizer=optimizer, loss=loss, metrics = ["accuracy"])
# print(model.summary())
model.fit(
    train_features,
    train_classes,
    validation_steps=(test_features,test_classes),
    batch_size=1024,
    epochs=500,
    verbose=2,
)


# #initialize embedding layer
# embedding_layer = Embedding(
#     vocab_size+1,
#     embed_size,
#     input_length=input_size,
#     weights=[embedding_weights]
# )
