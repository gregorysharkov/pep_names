import pandas as pd
from input_handler import word_embed_meta_data
from model import SiameseBiLSTM
from config import siamese_config

if __name__ == "__main__":
    path = "data\\combinations\\"
    true_data = pd.read_csv(path+"governors_true_match.csv",sep=";")
    false_data = pd.read_csv(path+"governors_false_match.csv",sep=";")

    combined_data = pd.concat([true_data,false_data])
    combined_data = combined_data.sample(frac=1,random_state=20210721)

    print(f"Combined dataset shape: {combined_data.shape}")

    original_names = list(combined_data.governor)
    alternative_names = list(combined_data.combinations)
    is_similar = list(combined_data.match)

    embedding_dim = siamese_config['EMBEDDING_DIM']
    tokenizer, embedding_matrix = word_embed_meta_data(original_names + alternative_names,  embedding_dim)

    embedding_meta_data = {
        'tokenizer': tokenizer,
	    'embedding_matrix': embedding_matrix
    }

    names_pair = [(x1,x2) for x1,x2 in zip(original_names,alternative_names)]
    del original_names
    del alternative_names

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

    # siamese = SiameseBiLSTM(CONFIG.embedding_dim, CONFIG.max_sequence_length, 
    #                         CONFIG.number_lstm_units , CONFIG.number_dense_units, 
    #                         CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, 
    #                         CONFIG.activation_function, CONFIG.validation_split_ratio)

    # best_model_path = siamese.train_model(names_pair, is_similar, embedding_meta_data, model_save_directory='./')