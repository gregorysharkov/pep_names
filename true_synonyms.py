import re
import json
import random
#from random import sample
#import pandas as pd
import unicodedata as ud
from itertools import permutations
from utils import load_data

latin_letters= {}
def is_latin(uchr):
    '''Checks if the character is lating
    '''
    try: return latin_letters[uchr]
    except KeyError:
         return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))


def only_roman_chars(unistr):
    '''Cheks if the string contains only roman characters
    '''
    return all(is_latin(uchr)
           for uchr in unistr
           if uchr.isalpha()) # isalpha suggested by John Machin


def get_synonyms(key,dict,only_roman=True):
    '''Function creates a list of all synonyms of a given key
    the key is also included into the final list

    Args:
        key: a string to be searched in the dict
        dict: the dictionary of synonyms
        only_roman: flag indicating if we want to deal only with roman characters

    Returns:
        a list containing the key and all its synonyms
    '''
    #add the name and its synonyms into the same list
    try:
        return_list = list([key] + dict[key])
    except KeyError:
        return_list = list(key)

    #some cleaning (to lower, remove duplicates, and keep only roman characters)
    return_list = [x.lower() for x in return_list]
    if only_roman:
        return_list = [x for x in return_list if only_roman_chars(x)]
    return_list.sort()
    return_list = [key] + return_list
    return return_list


def sample_permutations(name_list,combined_dict,n_samples,n_permutations):
    '''Function takes every word in the list of word in a name,
    searches for synonyms for each of them, samples one, then 
    generates several permutations of the selected words and 
    combines them into a string.
    The whole process is repeated n_samples times.
    The final number of strings generated is n_samples * n_permutations

    Args:
        name_list: list containing words, like ["grigory","joel"]
        combined_dict: dictionary of synonyms
        n_samples: number of different synonyms to be used for each word
        n_permutations: number of permutations to be taken from all combinations

    Return:
        list of strings like: ["grigory joel","joel grigory","grigorij joe"...]
    '''
    synonyms = []
    if name_list:
        for x in name_list:
            synonyms.append(get_synonyms(x,combined_dict))

    final_selection = []
    for _ in range(n_samples):
        if len(synonyms) == 0:
            selected_synonyms = []
        else: 
            selected_synonyms = [random.sample(x,1)[0] for x in synonyms]
        
        selected_synonyms_permutations = list(map(list,permutations(selected_synonyms)))
        if len(selected_synonyms_permutations) < n_permutations:
            selected_combinations = selected_synonyms_permutations
        else:
            selected_combinations = random.sample(selected_synonyms_permutations,n_permutations)

        selected_strings = [" ".join(x) for x in selected_combinations]
        selected_strings = [x.lower() for x in selected_strings]
        final_selection = list(set([*final_selection,*selected_strings]))

    return final_selection


def generate_combinations(name,combined_dict,n_samples,n_permutations):
    '''Function generates permutations between name and surname

    Args:
        name: list of words that a name contains
        combined_dict: dictionary of synonyms
        n_samples: number of different synonyms to be used for each word
        n_permutations: number of permutations to be taken from all combinations

    Returns:
        list of combined strings created based on permutations of the name and the surname
    '''
    #generate samples
    names = sample_permutations(name,combined_dict,n_samples,n_permutations)
    
    #add name and the inverse name in the samples list
    names.append(" ".join(name).lower())
    names.append(" ".join(name[::-1]).lower())

    return names


def generate_output(original,combined_dict,n_samples,n_permutations):
    '''Function processes a dataset with combined dictionary

    Args:
        original: the dataset to be processed. Should contain columns bg_name, name_split and surname_split
        combined_dict: dictionary of synonyms
        n_samples: number of different synonyms to be used for each word
        n_permutations: number of permutations to be taken from all combinations

    Returns:
        a dataset containing 3 columns:
            * bg_name: original name
            * combinations: one of the combinations of the original name
            * match: equal to 1 since these permutations are a true match (for training purposes)
    '''
    combinations = []
    for name in original.governor_split:
        combination = generate_combinations(name,combined_dict,n_samples,n_permutations)
        combinations.append(combination)

    original["combinations"] = combinations
    output_dataset = original.\
        explode("combinations").\
        drop_duplicates(subset=["governor","combinations"]).\
        sort_values(by=["governor","combinations"])\
        [["governor","combinations"]]
    
    output_dataset["match"] = 1
    return output_dataset


if __name__ == "__main__":
    path = "data\\dict\\"
    with open(path+"names.json", encoding="utf-8") as f:
        names = json.load(f)

    combined_dict = names

    original = load_data("data\\source\\united_states_governors.csv")

    n_samples = 10
    n_permutations = 30

    output_dataset = generate_output(original,combined_dict,n_samples,n_permutations) 
    output_dataset.governor = output_dataset.governor.str.lower()
    print(output_dataset.head(30))
    print(output_dataset.shape)

    output_dataset.to_csv(f"data\\combinations\\governors_true_match.csv",sep=";",header=True,index=False)
