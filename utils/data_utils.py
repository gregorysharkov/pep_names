import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

def clean_data(data):
    """Function cleans a dataset"""
    data = data.dropna()
    data = data.replace("\xa0"," ")
    return data

def load_dataset(path,col_names=["governor","combination","match"],limit=None):
    '''Function loads a dataset'''
    dataset = pd.read_csv(path,sep=";")
    dataset = clean_data(dataset)
    if limit:
        dataset = dataset[:limit]
    dataset.columns = col_names

    return dataset

def load_data(source_path,limit=None,debug=False,balance=True):
    '''Function loads true and false datasets'''
    true_data = load_dataset(source_path+"true_match.csv",limit=limit)
    print(f"{len(true_data)}")
    false_data = load_dataset(source_path+"false_match.csv", limit=limit)#.sample(n=len(true_data),random_state=20210924)
    if balance:
        false_data = false_data.sample(n=len(true_data),replace=True,random_state=20210924)
    print(f"{len(false_data)}")
    combined_data = pd.concat([true_data,false_data]).\
        sample(frac=1,random_state=20210924)
    
    if debug:
        print(combined_data.head(10))
    return combined_data

def train_test_split_pandas(data:pd.DataFrame,p_train,p_val):
    '''function splits the data'''
    n_train = int(p_train*len(data))
    n_val = int(p_val*len(data))
    n_test = len(data) - n_train - n_val

    raw_training = data[:n_train]
    raw_validation = data[n_train:n_train+n_val]
    raw_test = data[n_train+n_val:]

    return raw_training,raw_validation,raw_test


def save_tf_dataset(data,path):
    '''Function saves the tf dataset into the given folder'''
    tf.data.experimental.save(data,path,compression="GZIP")
    with open(path+"/element_spec","wb") as out_:
        pickle.dump(data.element_spec,out_)

def load_tf_dataset(path):
    """Function loads a tf dataset from a given location"""
    with open(path+"/element_spec","rb") as in_:
        element_spec = pickle.load(in_)    
    data = tf.data.experimental.load(path, element_spec, compression="GZIP")

    return data

def load_train_test(path):
    train = load_tf_dataset(path+"train")
    val = load_tf_dataset(path+"val")
    test = load_tf_dataset(path+"test")

    return train,val,test