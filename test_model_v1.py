import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Iterable

from textdistance import jaro
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data_utils import load_data, save_tf_dataset, load_tf_dataset 
from tokenizer.tokenizer import load_tokenizer
from model.v1.model import OuterModel, InnerModel
from model.v1.experiment_factory import BASE_EXPERIMENT, ABS_EXPERIMENT

from model.v1.model_utils import load_model_from_checkpoint
from model.v1.model_settings import ExperimentSettings

tf.config.run_functions_eagerly(False)

def preprocess_list(lst:Iterable, tokenizer:Tokenizer, max_words:int=None, max_char:int=None,debug=False):
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

    #some cleaning
    lst = list(map(lambda x: x.replace(u"\xa0"," "),lst))
    lst = list(map(lambda x: x.replace("  "," "),lst))
    lst = list(map(lambda x: x.replace("  "," "),lst))
    lst = list(map(lambda x: x.strip(),lst))

    #now let's convert every string into a matrix
    padded_test_string = [x + " "*(max_words-len(x.split())) for x in lst]
    test_split = [x.split(" ")[:max_words] for x in padded_test_string]
    test_sequences = [tokenizer.texts_to_sequences(x) for x in test_split]
    padded_test_sequences = [pad_sequences(x,maxlen=max_char,padding="post") for x in test_sequences]
    return_matrix = tf.data.Dataset.from_tensor_slices(padded_test_sequences)

    if debug:
        print(f"{padded_test_string=}, {np.shape(padded_test_string)=}")
        print(f"{test_split=}, {np.shape(test_split)=}")
        print(f"{test_sequences=} {np.shape(test_sequences)=}")
        print(f"{padded_test_sequences=} {np.shape(padded_test_sequences)=}")
        print(f"{return_matrix=}")

    return return_matrix

def construct_features(data,tokenizer,max_words=None,max_char=None,batch_size=None):
    """
    Function constructs tensorflow datasets from pandas dataset

    Args:
        data: pandas dataframe (should contain name, combination and match columns)
        tokenizer: tokenizer used to preprocess lists
        max_words: number of words, the string will be padded to
        max_chars: number of characters each word will be padded to
        match_size: size of the batch of the output dataset
    """
    names = preprocess_list(data.name,tokenizer,max_words,max_char,False)
    combinations = preprocess_list(data.combination,tokenizer,max_words,max_char,False)

    features = tf.data.Dataset.zip((names,combinations))
    match = tf.data.Dataset.from_tensor_slices(np.expand_dims(np.array(data.match),1))
    full_data = tf.data.Dataset.zip((features,match))

    if batch_size is not None:
        features = features.batch(batch_size)
        match = match.batch(batch_size)
        full_data = full_data.batch(batch_size)

    return features,match,full_data

def get_data(save_path,limit,tokenizer,experiment_settings,refresh=False,balance=True):
    """
    Function loads data
    
    Args:
        save_path : path used to read / write data
        limit : maximum number of rows to be loaded
        tokenizer: tokenizer used to convert strings into sequences
        experiment_settings: an instance of ExperimentSettings class
        refresh: a boolean idicating whether we need to refresh already stored data
                 If the data is not refreshed, it is read from the save_path, otherwise, 
                 new data is generated and written to the save_path location
        balance: a boolean indicating whether the data needs to be balanced (equal number of true and false matches)
    """
    if refresh:
        #load data
        data_path = "name_similarity/data/combinations/"
        raw_data = load_data(data_path,limit,balance)

        #train_test_split
        raw_train, raw_val = train_test_split(raw_data,train_size=.7)
        train_features,train_match,train  = construct_features(
            raw_train,
            tokenizer,
            experiment_settings.outer_settings.inner_settings.n_words,
            experiment_settings.outer_settings.inner_settings.n_characters,
            experiment_settings.fit_settings.batch_size)
        val_features,val_match,val = construct_features(
            raw_train,
            tokenizer,
            experiment_settings.outer_settings.inner_settings.n_words,
            experiment_settings.outer_settings.inner_settings.n_characters,
            experiment_settings.fit_settings.batch_size)

        save_tf_dataset(train,save_path+"train/train")
        save_tf_dataset(train_features,save_path+"train/train_features")
        save_tf_dataset(train_match,save_path+"train/train_match")

        save_tf_dataset(val,save_path+"val/val")
        save_tf_dataset(val_features,save_path+"val/val_features")
        save_tf_dataset(val_match,save_path+"val/val_match")
    else:
        train = load_tf_dataset(save_path+"train/train")
        train_features = load_tf_dataset(save_path+"train/train_features")
        train_match = load_tf_dataset(save_path+"train/train_match")

        val = load_tf_dataset(save_path+"val/val")
        val_features = load_tf_dataset(save_path+"val/val_features")
        val_match = load_tf_dataset(save_path+"val/val_match")

    return (train,train_features,val_match), (val,val_features,val_match)

def run_test(tokenizer:Tokenizer, experiment_settings: ExperimentSettings, checkpoint_path: str = None):
    """
    Basic function to demonstrate the model

    Args:
        tokenizer: tokenizer to be used to turn strings into matrices
        experiment_settings: an instance of ExperimentSettings
        checkpoint_path: path to the weihts of a model that has already been trained. If none, a new model will be generated
    """
    test_string = ["Grigory Sharkov", "Sharkov Grigory", "Boris Jonson","Bill Clinton","Bill Gates"]
    test_combinations = ["Grigory Sharkov", "Grigory Sharkov", "Moris Jonson","George Washington","William Gates"]
    test_match = [1,1,0,0,1]

    names = preprocess_list(
        lst = test_string,
        tokenizer = tokenizer,
        max_words=experiment_settings.outer_settings.inner_settings.n_words,
        max_char=experiment_settings.outer_settings.inner_settings.n_characters,
        debug=False)

    combinations = preprocess_list(
        lst = test_combinations,
        tokenizer = tokenizer,
        max_words=experiment_settings.outer_settings.inner_settings.n_words,
        max_char=experiment_settings.outer_settings.inner_settings.n_characters,
        debug=False)

    match = tf.data.Dataset.from_tensor_slices(np.array(test_match))

    features = tf.data.Dataset.zip(((names,combinations),)).batch(experiment_settings.fit_settings.batch_size)
    training_data = tf.data.Dataset.zip((features,match)).batch(experiment_settings.fit_settings.batch_size)

    if checkpoint_path:
        model = load_model_from_checkpoint(checkpoint_path,experiment_settings)
    else:
        model = OuterModel(experiment_settings.outer_settings)
        model.compile(optimizer="Nadam",loss="binary_crossentropy")

    similarities = model.predict(features)
    for name, combi, simi in zip(test_string,test_combinations,similarities):
        print(f"Comaring: '{name}' and '{combi}': \t{simi:.4f}")

def run_on_the_real_data(tokenizer:Tokenizer, experiment_settings: ExperimentSettings, limit:int, batch_size=None,refresh=False):
    """
    Fits a new model
    """
    save_path = "name_similarity/data/tokenized/20211019/"
    (train,train_features,val_match), (val,val_features,val_match) = get_data(save_path, limit, tokenizer, experiment_settings, refresh=refresh, balance=True)

    #model setup and train
    model = OuterModel(experiment_settings.outer_settings)
    model.compile(optimizer=experiment_settings.outer_settings.optimizer,
                  loss=experiment_settings.outer_settings.loss,
                  metrics=experiment_settings.outer_settings.metrics)
    model.fit(train,
              validation_data = val,
              epochs=experiment_settings.fit_settings.epochs,
              verbose=experiment_settings.fit_settings.verbose,
              callbacks=experiment_settings.fit_settings.callbacks)

    return model

def turn_matrix_into_strings(df, tokenizer):
    """
    A helper function that turns a tensorflow dataset into a list of strings
    """
    updated_dict = tokenizer.index_word
    updated_dict.update({0:"_"})
    
    def get_char(id):
        return updated_dict[id]

    def get_word(word):
        return "".join(get_char(char) for char in word)

    def clean_string(string):
        string = re.sub("_","",string)
        string = re.sub(u"\xa0"," ",string)
        string = re.sub("  ","",string)
        string = re.sub(r" ^",r"^",string)
        return string

    strings = []
    for el in df.as_numpy_iterator():
        string = " ".join(get_word(word) for word in el)
        strings.append(string)
    strings = [clean_string(x) for x in strings]
    return strings

def restore_experiment(checkpoint_path, experiment_settings):
    """
    Function restores an experiment, generates evaluation dataset and generates an distribution plot of predictions and true labels

    Args:
        checkpoint_path: path to the checkpoint to be loaded (the lates checkpoint is loaded)
        experiment_settings: an instance of experiment settings to be used
    """
    tokenizer_path = "name_similarity/data/tokenizer/20211005_tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)

    model = load_model_from_checkpoint(checkpoint_path, experiment_settings)
    
    save_path = "name_similarity/data/tokenized/20211019/"
    (train,train_features,val_match), (val,val_features,val_match) = get_data(save_path, None, tokenizer, experiment_settings, refresh=False, balance=True)
    val_names = turn_matrix_into_strings(val_features.map(lambda x,y: x).unbatch(),tokenizer)
    val_combi = turn_matrix_into_strings(val_features.map(lambda x,y: y).unbatch(),tokenizer)
    y_true = np.array([x[0] for x in val_match.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)).as_numpy_iterator()])
    y_pred = model.predict(val)
    jaro_dist = [jaro.similarity(name,comb) for name,comb in zip(val_names,val_combi)]

    df_pred = pd.DataFrame({"name":val_names,"alternative":val_combi,"y_pred":y_pred,"y_true":y_true, "jaro_dist":jaro_dist})
    df_pred.to_csv("name_similarity/evaluation.csv",sep=";",index=False)
    print(df_pred.head())

    df_pred.hist(column="y_pred",by="y_true")
    plt.xlabel("Similarity measure")
    plt.ylabel("Number of predictions")
    plt.savefig("name_similarity/images/plot_with_abs.png",dpi=300)

def main():
    tokenizer_path = "name_similarity/data/tokenizer/20211005_tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)

    experiment_settings = ABS_EXPERIMENT

    checkpoint_path = "/data/lab/cple/emb/grigory/name_similarity/experiments_with_two_layers/with_abs/20211019-150729/weights/"
    run_test(tokenizer, experiment_settings, checkpoint_path)
    # run_on_the_real_data(tokenizer,experiment_settings,None,1000,refresh=False)
    # restore_experiment(checkpoint_path, experiment_settings)


if __name__=="__main__":
    main()
