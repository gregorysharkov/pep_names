import random
from tqdm import tqdm
import pandas as pd
from utils.utils import load_dict, load_data
from true_combination_generator.true_combination_generator import TrueSynonymsGenerator
from utils.generation_utils import abbreviate_random_word, remove_random_word


def generate_true_match(name_list:list[str],name_dict:dict,n_samples:int=None,n_samples_from_dict:int=None,frac:float=.5) -> pd.DataFrame:
    '''
    Function generates a dataframe with true matches
    
    Args
    '''
    combinations = []
    pbar = tqdm(total = len(name_list))
    for name in name_list:
        generator = TrueSynonymsGenerator(name,n_sample_from_dict=n_samples_from_dict)
        generator.fit(name_dict)
        combinations.append(generator.sample(n_samples))
        pbar.update(n=1)

    return_df = pd.DataFrame({
        "name" : name_list,
        "combination": combinations,
        "match": [1]*len(name_list)
    }).explode(column="combination", ignore_index=True)

    random.seed(20211028)
    abbreviations_sample = return_df.sample(frac=frac)
    abbreviations_sample.combination.apply(abbreviate_random_word)
    
    shorts_sample = return_df.sample(frac=frac)
    shorts_sample.combination.apply(remove_random_word)

    return_df = pd.concat([return_df,abbreviations_sample,shorts_sample])
    return return_df


def main():
    dict_path = "data\\dict\\dictionary_en_fr_ar_ru.yml"
    name_dict = load_dict(dict_path)
    print(f"name dictionary length: {len(name_dict)}")

    data_path = "data\\source\\united_states_governors.csv"
    raw_data = load_data(data_path,"governor",sep=",")
    print(f"the name list length: {len(raw_data.governor)}")

    output_path = "data\\combinations\\"
    true_combinations = generate_true_match(name_list=raw_data.governor, name_dict=name_dict, n_samples=50, n_samples_from_dict=30)
    print(f"Length of true_combinations: {len(true_combinations)}")
    true_combinations.to_csv(output_path+"true_match.csv",sep=";",index=False)
    pass


if __name__ == "__main__":
    main()