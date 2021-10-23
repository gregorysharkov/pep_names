import pandas as pd
from utils.utils import load_dict, load_data
from true_combination_generator.true_combination_generator import TrueSynonymsGenerator


def generate_true_match(name_list,name_dict,n_samples=None):
    '''Function generates a dataframe with true matches'''
    combinations = []
    for name in name_list:
        generator = TrueSynonymsGenerator(name)
        generator.fit(name_dict)
        combinations.append(generator.sample(n_samples))

    return_df = pd.DataFrame({
        "name" : name_list,
        "combination": combinations,
        "match": [1]*len(name_list)
    }).explode(column="combination", ignore_index=True)

    return return_df


def main():
    dict_path = "name_similarity/data/dict/dictionary_en_fr_ar_ru.yml"
    name_dict = load_dict(dict_path)
    print(f"name dictionary length: {len(name_dict)}")

    data_path = "name_similarity/data/raw/bg_names.csv"
    raw_data = load_data(data_path,"bg_name")
    print(f"the name list length: {len(raw_data.bg_name)}")

    output_path = "name_similarity/data/combinations/"
    true_combinations = generate_true_match(raw_data.bg_name,name_dict,150)
    print(f"Length of true_combinations: {len(true_combinations)}")
    true_combinations.to_csv(output_path+"true_match.csv",sep=";",index=False)
    pass


if __name__ == "__main__":
    main()