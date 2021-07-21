from utils import load_data

def generate_false_permutations(name_split,data):
    '''function generates a of false matches for each line in the data dataset
    
    Args:
        name_split: a list containing list of names to be split
        data: dataset to be used to search for matches

    Returns:
        list that contains all element from data that contain at least
    '''
    #remove the requested name_split from the dataset
    data = data[data.governor!= " ".join(name_split)]
    #generate list
    filtered_data = data[data.governor_split.apply(lambda x: any(y in x for y in name_split))].governor.to_list()
    return filtered_data


if __name__ == "__main__":
    original = load_data("data\\source\\united_states_governors.csv").sort_values("governor")
    #generate combinations
    original["combinations"] = original.governor_split.apply(lambda x: generate_false_permutations(x, original))
    original = original.explode("combinations")
    #add match column
    original["match"] = 0
    #drop empty values
    original = original.dropna()
    original = original[["governor","combinations","match"]]
    #save data
    original.to_csv("data\\combinations\\governors_false_match.csv",sep=";",index=False)
    print(original.head(30))
    print(original.shape)