import json
import string
import pandas as pd

def capitalize_every_word(string):
    if " " in string:
        string_split = string.split(" ")
        capitalized_string = " ".join([x.capitalize() for x in string_split])
        return capitalized_string
    else:
        return string

def load_data(path,column,debug=True):
    '''Function loads data from the given path'''
    data = pd.read_csv(path,sep=";")
    data[column] = data[column].str.strip()
    data[column] = data[column].str.replace("  "," ")
    data[column] = data[column].str.translate(str.maketrans('', '', string.punctuation))
    data = data[[column]].drop_duplicates()
    data = data[[column]].dropna()
    data[column] = data[column].apply(capitalize_every_word)
    data[column+"_split"] = data[column].str.split(" ")
    return data

def load_dict(path):
    """Function loads the dictionary from a given path"""
    with open(path,"r") as f:
        contents = f.read()
        f.close()
    
    return_dict = json.loads(contents)
    return return_dict
