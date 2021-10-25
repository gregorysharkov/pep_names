from true_combinations_generator import TrueSynonymsGenerator

test_name = "Bill Gates"
test_dict = {"Bill": ["William","Guillaume"]}

test_gen = TrueSynonymsGenerator(test_name)
test_gen.fit(test_dict)
test_gen.sample()
print(f"Number of all possible combinations: {len(test_gen.combinations)}")
print(f"Number of selected combinations {len(test_gen.selected_combinations)}")
print(repr(test_gen))

