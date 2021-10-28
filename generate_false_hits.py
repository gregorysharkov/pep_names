import random
import pandas as pd
from utils.utils import load_data
from utils.generation_utils import abbreviate_random_word, remove_random_word

def generate_false_match(name_split:list[str],data:pd.DataFrame,col_name:str)->list:
    '''function generates a of false matches for a given list of words
    
    Args:
        name_split: a list containing list of names to be split
        data: dataset to be used to search for matches
        col_name: column name used for text match
    Returns:
        list that contains all element from data that contain at least
    '''
    #remove the requested name_split from the dataset
    data = data[data[col_name]!= " ".join(name_split)]
    #generate list
    filter_condition = data[col_name+"_split"].apply(lambda x: any(y in x for y in name_split))
    filtered_data = data[filter_condition][col_name].to_list()
    return filtered_data

        
def process_dataset(data:pd.DataFrame,col_name:str,frac:float=.6)->pd.DataFrame:
    '''Function processes dataset by generating false combinations for every line'''
    #generate combination for every line
    data = data.dropna()
    data["combinations"] = data[col_name+"_split"].apply(lambda x: generate_false_match(x,data,col_name))

    #add match column:
    data["match"] = 0

    #explode the data
    data = data.explode(column="combinations")

    #generate shorts and abbreviations
    random.seed(20211028)
    abbreviations_sample = data.sample(frac=frac)
    abbreviations_sample.combinations.apply(abbreviate_random_word)
    
    shorts_sample = data.sample(frac=frac)
    shorts_sample.combinations.apply(remove_random_word)

    data = pd.concat([data,abbreviations_sample,shorts_sample])

    #drop empty values and do some cleaning
    data = data.dropna()
    data = data[[col_name,"combinations","match"]]
    data = data.drop_duplicates()

    return data


def main():
    data_path = "data\\source\\united_states_governors.csv"
    col_name = "governor"
    raw_data = load_data(data_path,col_name,sep=",")
    print(f"Name list length: {len(raw_data.governor)}")

    false_match_data = process_dataset(raw_data,col_name)
    print(f"Synonyms list length: {len(false_match_data.governor)}")
    
    output_path = "data\\combinations\\"
    false_match_data.to_csv(output_path+"false_match.csv",sep=";",index=False)
    pass

if __name__=="__main__":
    main()