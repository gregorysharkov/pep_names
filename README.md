# pep_names

The goal of this prohect is to build an deep learning model that measures similarity of names of two individuals. I am doing this for my project at work, but I do not want to share the data that we are using, so for this repo I will prepare my own dataset using [Data from Open ICPSR](https://www.openicpsr.org/openicpsr/project/102000/version/V3/view). This is a list of all governors of all states in the USA.  
  
The model has to be able to handle misspellings, shold tolerate some alternations in the words order, for example "first name" + "second name" should be considered the same as "second name" + "first name" and ideally should be able to handle different writings of the same name, for example Bill = William or Stéphane = Etien.
  
So I will try to build a deep learning network described by [Chen Zhao and Yeye He](https://www.microsoft.com/en-us/research/uploads/prod/2019/04/Auto-EM.pdf) in tensorflow.

## Preparing the data
The first step is to prepare the data. I will need to generate two sets:

1. positive matches. Pairs of names that represent the same person, therefore the distance should be 1
2. negative matches. Pairs of names that do not represent the same person, therefore the distance should be 0.
  
To generate positive matches I build a module that uses [wikidata api](https://www.wikidata.org/wiki/Wikidata:Data_access). I have built a class named "Synonym_finder" that given a string will try to search alternative spellings of this string asumming that this is a name or a surname. So I create a dictionary that fetches every word in my dataset to create a list of synonyms of each word i try to find.  
  
The next step is to create positive matches by using the original data and this dictionary. For each word in a name, we will try to replace it with a synonym and use one of the synonyms instead and then permutating word order in the given name.
  

## Seameese network
The main idea is that we need a network that will be generating some representation of a string. We will feet two strings into this network, pass them both though this network, obtain two representations (one for each string) and them compare them with some measure. The final prediction will be a linear function of this similarity. The overall architecture is presented in the figure below:  
![model_architecture](./images/model_architecture.jpg)
  
I chose the following architecture for the inner model:
![inner_model_architecture](./images/inner_model.jpg)

As a base model I will be using the approach proposed [here](https://github.com/amansrivastava17/lstm-siamese-text-similarity)

## Usage
The model has been pretrained with settings from `install_settings.py`, we can use the model with the following code:
```python
import pandas as pd
import numpy as np
from tokenizer import load_tokenizer
from instance_settings import inner_settings_1, outer_settings_1
from utils_model import load_model_from_checkpoint,compare_strings

np.set_printoptions(precision=4)
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
```
Output:
```
Comparing 'Boris Jonson' and 'Borya Jonson'
Similarity: 0.9861
Prediction: 0.6604 => 1.0

Comparing 'Moris Jonson' and 'Boris Jonson'
Similarity: 0.5186
Prediction: 0.4860 => 0.0
```