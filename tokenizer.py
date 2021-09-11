import io
import json
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import (
    Tokenizer, 
    tokenizer_from_json
)


def train_tokenizer(lst) -> Tokenizer:
    """
    Function trains a tokenizer based on provided list of strings

    Args:
        lst: list of strings to be tokenized
    
    Returns:
        tokenizer trained on the given data
    """
    tk = Tokenizer(num_words=None, char_level=True, oov_token="UNK")
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

def main():
    path = "data\\combinations\\"
    true_data = pd.read_csv(path+"governors_true_match.csv",sep=";")
    false_data = pd.read_csv(path+"governors_false_match.csv",sep=";")
    combined_data = pd.concat([true_data,false_data])
    combined_data = combined_data.sample(frac=1,random_state=20210826)

    governors_list = list(combined_data.governor)
    combination_list = list(combined_data.combinations)

    tokenizer = train_tokenizer(governors_list+combination_list)
    save_tokenizer(
        tokenizer,
        "output_model\\architecture_with_abs\\tokenizer.json"
    )

    tokenizer = load_tokenizer("output_model\\architecture_with_abs\\tokenizer.json")
    print(tokenizer.word_index)
    pass

if __name__ == '__main__':
    main()
