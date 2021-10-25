import re
import io
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Iterable
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

def preprocess_list_into_matrix(lst,tokenizer,max_words=None,max_char=None,debug=False):
    """Function turns a list of strings into a list of 2x2 matrices: row for word, col for character"""
    lst = list(map(lambda x: x.replace(u"\xa0"," "),lst))
    lst = list(map(lambda x: x.replace("-",""),lst))
    lst = list(map(lambda x: x.replace(",",""),lst))
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
    word_split = [x.split(" ",maxsplit=max_words-1) for x in padded_string]
    
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
        string = re.sub("-","",string)
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