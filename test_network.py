import os
import datetime as datetime
import tensorflow as tf
import pandas as pd
import numpy as np

from tokenizer import (
    load_tokenizer, 
    preprocess_list
)

from model_settings import (
    InnerModelSettings, 
    OuterModelSettings, 
    FitSettings
)

from utils_model import (
    create_model,
    load_model_from_checkpoint,
    compare_representations,
    compare_strings,
)

np.set_printoptions(precision=4)

#load data
path = "data\\combinations\\"
true_data = pd.read_csv(path+"governors_true_match.csv",sep=";")
false_data = pd.read_csv(path+"governors_false_match.csv",sep=";")
combined_data = pd.concat([true_data,false_data])
combined_data = combined_data.sample(frac=1,random_state=20210826)

governors_list = list(combined_data.governor)
combination_list = list(combined_data.combinations)
match = list(combined_data.match)

#load the tokenizer
tk = load_tokenizer("output_model\\architecture_with_abs\\tokenizer.json")

#set up settings
inner_settings_1 = InnerModelSettings(
    input_embedding = 129,
    n_embedding_dims = 512,
    n_gru = 40,
    n_dense = 80,
    n_units_attention=40
)

outer_settings_1 = OuterModelSettings(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(1e-4),
    metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy"),
               tf.keras.metrics.Precision(name="precision")]
)

fit_settings_1 = FitSettings(
    batch_size = 1000,
    epochs = 10,
    verbose=2,
    callbacks=[]
)

#load model
model = load_model_from_checkpoint(
    "output_model\\architecture_with_abs\\20210906-221827\\weights\\",
    inner_settings_1,
    outer_settings_1
)

#initialize and preprocess strings
my_test = ["Boris Jonson","Borya Jonson"]
compare_strings(my_test,tk,model,debug=True,give_representations=False,echo=False)