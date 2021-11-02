import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from textdistance import jaro
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer

from utils.data_utils import load_data, save_tf_dataset, load_tf_dataset 
from tokenizer.tokenizer import load_tokenizer, preprocess_list_into_matrix, turn_matrix_into_strings
from model.v1.model import OuterModel
from model.v1.experiment_factory import BASE_EXPERIMENT, ABS_EXPERIMENT

from model.v1.model_utils import load_model_from_checkpoint
from model.v1.model_settings import ExperimentSettings

tf.config.run_functions_eagerly(False) 

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
    names = preprocess_list_into_matrix(data.governor,tokenizer,max_words,max_char,False)
    combinations = preprocess_list_into_matrix(data.combination,tokenizer,max_words,max_char,False)

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
        data_path = "data\\combinations\\"
        raw_data = load_data(data_path,limit,balance)

        #train_test_split
        raw_train, raw_val = train_test_split(raw_data,train_size=.7)
        train_features,train_match,train  = construct_features(
            data = raw_train,
            tokenizer = tokenizer,
            max_words = experiment_settings.outer_settings.inner_settings.n_words,
            max_char = experiment_settings.outer_settings.inner_settings.n_characters,
            batch_size = None)#experiment_settings.fit_settings.batch_size)
        val_features,val_match,val = construct_features(
            data = raw_train,
            tokenizer = tokenizer,
            max_words = experiment_settings.outer_settings.inner_settings.n_words,
            max_char = experiment_settings.outer_settings.inner_settings.n_characters,
            batch_size = None)#experiment_settings.fit_settings.batch_size)

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
    test_string = ["Grigory Sharkov", "Grigory Sharkov", "Sharkov Grigory", 
                   "Boris Jonson", "Bill Clinton", "Bill Gates", "Bill Gates"]
    test_combinations = ["Grigory Sharkov", "Grigory Sharkov", "Grigory Sharkov", 
                         "Moris Jonson", "George Washington", "William Gates", "Bill Clinton"]
    test_match = [1,1,1,0,0,1,0]

    names = preprocess_list_into_matrix(
        lst = test_string,
        tokenizer = tokenizer,
        max_words=experiment_settings.outer_settings.inner_settings.n_words,
        max_char=experiment_settings.outer_settings.inner_settings.n_characters,
        debug=False)

    combinations = preprocess_list_into_matrix(
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


def run_on_the_real_data(tokenizer:Tokenizer, experiment_settings: ExperimentSettings, limit:int, refresh=False,checkpoint_path=None):
    """
    Fits a new model
    """
    save_path = "data\\tokenized\\20211024\\"
    (train,train_features,val_match), (val,val_features,val_match) = get_data(save_path, limit, tokenizer, experiment_settings, refresh=refresh, balance=True)
    train = train.batch(experiment_settings.fit_settings.batch_size)
    val = val.batch(experiment_settings.fit_settings.batch_size)
    print(train.take(2))

    # for i, ((x, y),z) in enumerate(train):
    #     print(i, x.numpy().shape, y.numpy().shape)
    #model setup and train
    if checkpoint_path:
        model = load_model_from_checkpoint(checkpoint_path,experiment_settings)
    else:
        model = OuterModel(experiment_settings.outer_settings)
        model.compile(optimizer=experiment_settings.outer_settings.optimizer,
                    loss=experiment_settings.outer_settings.loss,
                    metrics=experiment_settings.outer_settings.metrics)
    model.fit(train,
              validation_data = val,
              batch_size = experiment_settings.fit_settings.batch_size,
              epochs=experiment_settings.fit_settings.epochs,
              verbose=experiment_settings.fit_settings.verbose,
              callbacks=experiment_settings.fit_settings.callbacks)

    print(model.summary())
    return model


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
    val_names = turn_matrix_into_strings(val_features.map(lambda x,_y: x).unbatch(),tokenizer)
    val_combi = turn_matrix_into_strings(val_features.map(lambda _x,y: y).unbatch(),tokenizer)
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
    tokenizer_path = "data\\tokenizer\\20211023_tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)

    experiment_settings = ABS_EXPERIMENT

    checkpoint_path = "logs\\baseline\\adadelta\\20211102-200019\\weights\\"
    run_on_the_real_data(
        tokenizer = tokenizer,
        experiment_settings = experiment_settings,
        limit=None,
        refresh=False,
        checkpoint_path=checkpoint_path)
    run_test(tokenizer, experiment_settings, checkpoint_path)
    # restore_experiment(checkpoint_path, experiment_settings)


if __name__=="__main__":
    main()
