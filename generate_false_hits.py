from utils.utils import load_data

def generate_false_match(name_split,data,col_name):
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

        
def process_dataset(data,col_name):
    '''Function processes dataset by generating false combinations for every line'''
    #generate combination for every line
    data["combinations"] = data[col_name+"_split"].apply(lambda x: generate_false_match(x,data,col_name))

    #add match column:
    data["match"] = 0

    #explode the data
    data = data.explode(column="combinations")

    #drop empty values
    data = data.dropna()
    data = data[[col_name,"combinations","match"]]

    return data


def main():
    data_path = "name_similarity/data/raw/bg_names.csv"
    col_name = "bg_name"
    raw_data = load_data(data_path,col_name)
    print(f"Name list length: {len(raw_data.bg_name)}")

    false_match_data = process_dataset(raw_data,"bg_name")
    print(f"Synonyms list length: {len(false_match_data.bg_name)}")
    
    output_path = "name_similarity/data/combinations/"
    false_match_data.to_csv(output_path+"false_match.csv",sep=";",index=False)
    pass

if __name__=="__main__":
    main()