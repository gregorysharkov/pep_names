import numpy as np
import tensorflow as tf
import datetime as datetime

from tensorflow.keras.preprocessing.text import Tokenizer

from model_settings import (
    InnerModelSettings, 
    OuterModelSettings, 
    FitSettings
)

from char_level_rnn_with_attention import OuterModel

from tokenizer import preprocess_list

def create_model(inner_settings:InnerModelSettings,outer_settings:OuterModelSettings) -> OuterModel:
    """
    function creates an OuterModel with provided settings.

    Args:
        inner_settings: an instannce of InnerModelSettings
        outer_settings: an instannce of OuterModelSettings
    """
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
              print_summary:bool=False) -> OuterModel:
    """
    Function trains an OuterModel with provided data and fit_settings

    Args:
        model: an instace of OuterModel
        train_data: data, the model will be trained on
        val_data: validation data
        fit_settings: an instance of FitSettings
        print_summary: a boolean indicating whether to print out the summary of the trained model
    """

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


def load_model_from_checkpoint(
    checkpoint_path: str,
    inner_settings: InnerModelSettings,
    outer_settings: OuterModelSettings) -> OuterModel:
    """
    function loads a model from a checkpoint

    Args: 
        checkpoint_path: 
        inner_settings: an instance of InnerModelSettings
        outer_model: an instance of OuterModelSettings

    Returns:
        an instance of OuterModel with loaded settings from checkpoint
    """
    checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    model = create_model(inner_settings,outer_settings)
    model.load_weights(checkpoint)
    return model


def compare_representations(
    input_a, 
    input_b, 
    model, 
    debug=False,
    give_representations=False,
    echo=False):
    """
    Function compares two sequences using a provided model

    Args:
        input_a: a numpy sequence to be fed into the model
        input_b: a numpy sequence to be fed into the model
        model: an instance of OuterModel
        debug: a boolean flag indigating whether to print the debug information
        give_representations: a boolean flag indigating whether to show the final representations
        echo: a boolean flag indigating whether the model should print the shapes of the elements
        passed through it.

    returns:
        cosine similarity of two representations
        a tuple containing both representations
        prediction of the model
    """
    prediction = model(
        (input_a.reshape(-1,len(input_a)),input_b.reshape(-1,len(input_b))),
        training=False,
        echo=echo
    )

    if debug:
        if give_representations:
            print(f"Representation of A: {model.repr_a}")
            print(f"Representation of B: {model.repr_b}")
        print(f"Similarity: {model.cosine_similarity[0]:.4f}")
        print(f"Prediction: {prediction[0][0]:.4f} => {np.round(prediction[0][0],0)}")


    return model.cosine_similarity, (model.repr_a,model.repr_a), np.round(prediction[0][0],4)


def compare_strings(string_pair: list, tokenizer: Tokenizer, model: OuterModel, **kwargs):
    """
    a wrapper for compare representations. takes strings, turns them into sequences and measures their similarity
    
    Args:
        string_pair: a list of two strings
        tokenizer: an instance of a Tokenizer
        model: an instance of OuterModel
        **kwargs: keword arguments to be passed further to compare representation function

    returns:
        output of the compare_representations function
    """
    seq = preprocess_list(string_pair, tokenizer)
    debug = False if "debug" in kwargs else kwargs["debug"]
    if debug:
        print(f"Comparing '{string_pair[0]}' and '{string_pair[1]}'")

    return compare_representations(seq[0],seq[1],model,**kwargs)