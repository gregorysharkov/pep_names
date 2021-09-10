import os
import datetime as datetime
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

np.set_printoptions(precision=4)

path = "data\\combinations\\"
true_data = pd.read_csv(path+"governors_true_match.csv",sep=";")
false_data = pd.read_csv(path+"governors_false_match.csv",sep=";")
combined_data = pd.concat([true_data,false_data])
combined_data = combined_data.sample(frac=1,random_state=20210826)
names = sorted(set(list(combined_data.governor) + list(combined_data.combinations)))
words = sorted(set(word for name in list(map(str.split,names)) for word in name))
vocab = sorted(set(character for word in words for character in word))

governors_list = list(combined_data.governor)
combination_list = list(combined_data.combinations)
match = list(combined_data.match)

tk = Tokenizer(num_words=None, char_level=True, oov_token="UNK")
tk.fit_on_texts(governors_list+combination_list)

def preprocess_list(lst,tokenizer,max_len=None):
    return_seq = tokenizer.texts_to_sequences(lst)
    return np.array(pad_sequences(return_seq, maxlen=max_len,padding="post"),dtype="float32")


from model_settings import InnerModelSettings, OuterModelSettings, FitSettings
from char_level_rnn_with_attention import OuterModel

def create_model(inner_settings:InnerModelSettings,outer_settings:OuterModelSettings):
    model = OuterModel(inner_settings)
    
    model.compile(
        loss= outer_settings.loss, 
        optimizer=outer_settings.optimizer,
        metrics=outer_settings.metrics,
    )

    return model

def fit_model(model: tf.keras.Model,
              train_data: tf.data.Dataset,
              val_data: tf.data.Dataset,
              fit_settings: FitSettings,
              print_summary:bool=False):

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"logs will be saved to: {log_dir}")
    checkpoint_path = log_dir + "/weights/cp-{epoch:02d}.ckpt"

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1
    )

    checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_path,
        verboise=1,
        save_weights_only=True,
        save_best_only=True
    )

    fit_settings.callbacks = [tensorboard_callback,checkpoints_callback]

    model.fit(
        train_data,
        batch_size = fit_settings.batch_size,
        epochs = fit_settings.epochs,
        validation_data = val_data,
        verbose=fit_settings.verbose,
        callbacks=fit_settings.callbacks
    )
    
    if print_summary:
        print(model.summary())
        
    return model

def compare_representations(input_a, input_b, model, debug=False,give_representations=False):
    outer_model = model
    prediction = outer_model(
        (input_a.reshape(-1,len(input_a)),input_b.reshape(-1,len(input_b))),
        False,True
    )

    if debug:
        if give_representations:
            print(f"Representation of A: {outer_model.repr_a}")
            print(f"Representation of B: {outer_model.repr_b}")
        print(f"Similarity: {outer_model.cosine_similarity[0]:.4f}")
        print(f"Prediction: {prediction[0][0]:.4f} => {np.round(prediction[0][0],0)}")


    return outer_model.cosine_similarity, (outer_model.repr_a,outer_model.repr_a)

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

latest_chkpt = tf.train.latest_checkpoint("output_model\\architecture_with_abs\\20210906-221827\\weights\\")
model = create_model(inner_settings_1,outer_settings_1)
model.load_weights(latest_chkpt)

my_test = ["Boris Jonson","Borya Jonson"]
my_test_seq = preprocess_list(my_test, tk)

print(f"Comparing '{my_test[0]}' and '{my_test[1]}'")
similarity, representations = compare_representations(
    my_test_seq[0],
    my_test_seq[1],
    model,
    True
)
