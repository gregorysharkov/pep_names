import pandas as pd
from input_handler_char import char_embed_meta_data
from model import SiameseBiLSTM
from config import siamese_config
from input_handler_char import Embedder

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

    emb = Embedder(original_names+alternative_names, embedding_dim)
    emb.set_tokenizer()
    print(emb.tokenizer.word_index)

#    tokenizer, embedding_matrix = char_embed_meta_data(original_names + alternative_names,  embedding_dim)

