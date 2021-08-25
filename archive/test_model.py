from operator import itemgetter
from keras.models import load_model
from config import siamese_config
from input_handler import create_test_data
from input_handler import word_embed_meta_data
import pandas as pd

path = "data\\combinations\\"
true_data = pd.read_csv(path+"governors_true_match.csv",sep=";")
false_data = pd.read_csv(path+"governors_false_match.csv",sep=";")

combined_data = pd.concat([true_data,false_data])
combined_data = combined_data.sample(frac=1,random_state=20210721)

print(f"Combined dataset shape: {combined_data.shape}")

original_names = list(combined_data.governor)
alternative_names = list(combined_data.combinations)
s_similar = list(combined_data.match)

embedding_dim = siamese_config['EMBEDDING_DIM']
tokenizer, embedding_matrix = word_embed_meta_data(original_names + alternative_names,  embedding_dim)


best_model_path = "checkpoints\\1627218633\\lstm_50_50_0.17_0.25.h5"
model = load_model(best_model_path)

class Configuration(object):
    """Dump stuff here"""
    
CONFIG = Configuration()

CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']

test_name_pairs = [('Grigory Sharkov','Grigory Sharkov'),
				   ('Sharkov Grigory','Grigory Sharkov'),
				   ('Julia Sharkova','Grigory Sharkov'),
				   ("Grigory Sharkov","Gregory Sharkov")]

test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_name_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])
preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
results = [(x, y, z) for (x, y), z in zip(test_name_pairs, preds)]
results.sort(key=itemgetter(2), reverse=True)
print(results)