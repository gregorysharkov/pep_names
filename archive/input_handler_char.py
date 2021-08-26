from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import numpy as np
import gc

class Embedder():
    documents: list[str]
    embedding_dim : int
    tokenizer: Tokenizer
#    char_vectors: Word2Vec

    def __init__(self,documents,embedding_dim):
        self.documents = documents
        self.embedding_dim = embedding_dim

    def set_tokenizer(self):
        documents = [[ch for ch in x.lower()] for x in self.documents]
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(documents)
        self.tokenizer = tokenizer

def train_char2vec(documents, embedding_dim):
    """
    train char2vector over traning documents
    Args:
        documents (list): list of document
        embedding_dim (int): outpu wordvector size
    Returns:
        word_vectors(dict): dict containing words and their respective vectors
    """
    model = Word2Vec(documents, min_count=1, vector_size=embedding_dim)
    word_vectors = model.wv
    del model
    return word_vectors


def create_embedding_matrix(tokenizer, word_vectors, embedding_dim):
    """
    Create embedding matrix containing word indexes and respective vectors from word vectors
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object containing word indexes
        word_vectors (dict): dict containing word and their respective vectors
        embedding_dim (int): dimention of word vector
    Returns:
    """
    nb_words = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    print("Embedding matrix shape: %s" % str(embedding_matrix.shape))
    for word, i in word_index.items():
        try:
            embedding_vector = word_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except KeyError:
            print("vector not found for word - %s" % word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix


def char_embed_meta_data(documents, embedding_dim):
    """
    Load tokenizer object for given vocabs list
    Args:
        documents (list): list of document
        embedding_dim (int): embedding dimension
    Returns:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        embedding_matrix (dict): dict with word_index and vector mapping
    """
    documents = [[ch for ch in x.lower()] for x in documents]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(documents)
    print(tokenizer.word_index)

    word_vector = train_word2vec(documents, embedding_dim)
    embedding_matrix = create_embedding_matrix(tokenizer, word_vector, embedding_dim)
    del word_vector
    gc.collect()
    return tokenizer, embedding_matrix


def create_train_dev_set(tokenizer, names_pair, is_similar, max_sequence_length, validation_split_ratio):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        names_pair (list): list of tuple of names pairs
        is_similar (list): list containing labels if respective names in name1 and name2
                           are same or not (1 if same else 0)
        max_sequence_length (int): max sequence length of names to apply padding
        validation_split_ratio (float): contain ratio to split training data into validation data
    Returns:
        train_data_1 (list): list of input features for training set from names1
        train_data_2 (list): list of input features for training set from names2
        labels_train (np.array): array containing similarity score for training data
        leaks_train(np.array): array of training leaks features
        val_data_1 (list): list of input features for validation set from names1
        val_data_2 (list): list of input features for validation set from names2
        labels_val (np.array): array containing similarity score for validation data
        leaks_val (np.array): array of validation leaks features
    """
    names1 = [x[0].lower() for x in names_pair]
    names2 = [x[1].lower() for x in names_pair]
    train_sequences_1 = tokenizer.texts_to_sequences(names1)
    train_sequences_2 = tokenizer.texts_to_sequences(names2)
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(train_sequences_1, train_sequences_2)]

    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=max_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=max_sequence_length)
    train_labels = np.array(is_similar)
    leaks = np.array(leaks)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

    del train_padded_data_1
    del train_padded_data_2
    gc.collect()

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]

    return train_data_1, train_data_2, labels_train, leaks_train, val_data_1, val_data_2, labels_val, leaks_val

def create_test_data(tokenizer, test_names_pair, max_sequence_length):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        test_names_pair (list): list of tuple of names pairs
        max_sequence_length (int): max sequence length of names to apply padding
    Returns:
        test_data_1 (list): list of input features for training set from names1
        test_data_2 (list): list of input features for training set from names2
    """
    test_names1 = [x[0].lower() for x in test_names_pair]
    test_names2 = [x[1].lower() for x in test_names_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_names1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_names2)
    leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                  for x1, x2 in zip(test_sequences_1, test_sequences_2)]

    leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)

    return test_data_1, test_data_2, leaks_test