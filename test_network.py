# import os
# import datetime as datetime
# import tensorflow as tf
import pandas as pd
import numpy as np
from tokenizer import load_tokenizer
from instance_settings import inner_settings_1, outer_settings_1
from utils_model import load_model_from_checkpoint,compare_strings

np.set_printoptions(precision=4)

# #load data
# path = "data\\combinations\\"
# true_data = pd.read_csv(path+"governors_true_match.csv",sep=";")
# false_data = pd.read_csv(path+"governors_false_match.csv",sep=";")
# combined_data = pd.concat([true_data,false_data])
# combined_data = combined_data.sample(frac=1,random_state=20210826)

# governors_list = list(combined_data.governor)
# combination_list = list(combined_data.combinations)
# match = list(combined_data.match)

#load the tokenizer
tk = load_tokenizer("output_model\\architecture_with_abs\\tokenizer.json")

#load model
model = load_model_from_checkpoint(
    "output_model\\architecture_with_abs\\20210906-221827\\weights\\",
    inner_settings_1,
    outer_settings_1
)

#initialize and preprocess strings
my_test = [
    ["Boris Jonson","Borya Jonson"],
    ["Moris Jonson", "Boris Jonson"]
]

for pair in my_test:
    compare_strings(pair,tk,model,debug=True,give_representations=False,echo=False)