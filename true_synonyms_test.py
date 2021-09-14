import json
#from true_synonyms import *
from true_combinations_generator import TrueSynonymsGenerator

test_name = "Bill,. Gates Geferson"
test_dict = {"Bill": ["William","Guillaume"]}

test_gen = TrueSynonymsGenerator(test_name)
test_gen.fit(test_dict)
test_gen.sample()
print(f"Number of all possible combinations: {len(test_gen.combinations)}")
print(f"Number of selected combinations {len(test_gen.selected_combinations)}")
print(repr(test_gen))
# path = "data\\dict\\"
# with open(path+"names.json", encoding="utf-8") as f:
#     names = json.load(f)

# #print(get_synonyms(key="Bill",dict=names,only_roman=True))
# print(sample_permutations(name_list=["Bill","Clinton"], combined_dict=names, n_samples=2, n_permutations=3))


