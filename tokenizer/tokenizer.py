import io
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

def train_tokenizer(lst) -> Tokenizer:
    """
    Function trains a tokenizer based on provided list of strings
    Args:
        lst: list of strings to be tokenized
    
    Returns:
        tokenizer trained on the given data
    """
    tk = Tokenizer(num_words=None, char_level=True, oov_token="_",lower=False)
    tk.fit_on_texts(lst)
    return tk


def load_tokenizer(path: str) -> Tokenizer:
    """
    function loads a tokenizer from a json file
    Args:
        path: path to the json file
    Returns:
        a tokenizer
    """
    with open(path) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    """
    Function saves tokenizer at a given path
    Args:
        tokenizer: an object to be saved
        path: relative path where the tokenizer will be saved
    Returns: None
    """
    tk_json = tokenizer.to_json()
    with io.open(path,"w",encoding="utf-8") as f:
        f.write(json.dumps(tk_json, ensure_ascii=False))
    return


def preprocess_list(lst,tokenizer,max_len=None):
    """
    function preprocesses a list of values returning tokenized sequences
    Args:
        lst: list of strings to be processed
        tokenizer: a tokenizer object
        max_len: if we need to ensure the same length of strings, we can provide an integer here
    Returns:
        a numpy array with tokenized sequences. Each sequence in a separate row
    """
    return_seq = tokenizer.texts_to_sequences(lst)
    seq = np.array(
        pad_sequences(return_seq, maxlen=max_len,padding="post"),
        dtype="float32"
    )
    return seq

def preprocess_list_into_matrix(lst,tokenizer,max_words=None,max_char=None,debug=False):
    """Function turns a list of strings into a list of 2x2 matrices: row for word, col for character"""
    lst = list(map(lambda x: x.replace(u"\xa0"," "),lst))
    lst = list(map(lambda x: x.replace("  "," "),lst))
    lst = list(map(lambda x: x.replace("  "," "),lst))
    lst = list(map(lambda x: x.strip(),lst))
    if not max_words:
        word_counts = [len(x.split()) for x in lst]
        max_words = max(word_counts)
    
    if not max_char:
        words = list(set([word for name in lst for word in name.split()]))
        max_char = max([len(word) for word in words])
    
    padded_string = [x + " _"*(max_words-len(x.split())) for x in lst]
    word_split = [x.split(" ",maxsplit=max_words) for x in padded_string]
    
    sequences = [tokenizer.texts_to_sequences(x) for x in word_split]    
    padded_sequences = [pad_sequences(x,maxlen=max_char,padding="post") for x in sequences]
    #debug trying to find incorrect shapes
    for i in range(len(padded_sequences)):
        if np.shape(padded_sequences[i]) != (max_words,max_char):
            print("****************************")
            print(f"Found an incorrect element: '{lst[i]}', shape: {np.shape(padded_sequences[i])}")
            print(f"\t its padded version: {padded_string[i]}")
            print(f"\t its word split: {word_split[i]}")
            print(f"\t its sequence: {sequences[i]}")
    
    final_output = tf.data.Dataset.from_tensor_slices(padded_sequences)

    if debug:
        print(f"{padded_string=}")
        print(f"{word_split=}")
        print(f"{sequences=}")
        print(f"{padded_sequences=}")
        print(f"{final_output=}")

    return final_output

def main():
    path = "data\\combinations\\"
    col_list = ["governor","combinations","match"]
    true_data = pd.read_csv(path+"governors_true_match.csv",sep=";").dropna()
    false_data = pd.read_csv(path+"governors_false_match.csv",sep=";").dropna()

    true_data.columns = col_list
    false_data.columns = col_list

    combined_data = pd.concat([true_data,false_data])
    combined_data = combined_data.sample(frac=1,random_state=20210826)

    name_list = list(combined_data.governor)
    combination_list = list(combined_data.combinations)

    print(name_list[:5])
    print(combination_list[:5])

    tokenizer = train_tokenizer(name_list+combination_list)
    save_tokenizer(
        tokenizer,
        "data\\tokenizer\\20211023_tokenizer.json"
    )

    tokenizer = load_tokenizer("data\\tokenizer\\20211023_tokenizer.json")
    print(tokenizer.word_index)
    pass

if __name__ == '__main__':
    main()