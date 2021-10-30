import datetime
import tensorflow as tf
from typing import List
from dataclasses import dataclass, field

@dataclass
class BiGruSettings:
    name : str
    input_shape: tuple[int]
    n_units: int

@dataclass 
class DistanceSettings:
    name : str
    mode: str

@dataclass
class InnerModelSettings:
    name: str
    n_words : int
    n_characters: int
    embedding_input_dim : int
    embedding_output_dim : int
    n_char_rnn_units: int
    n_word_rnn_units: int
    char_level_settings: BiGruSettings = None
    word_level_settings: BiGruSettings = None

    def __post_init__(self):
        self.char_level_settings = BiGruSettings(
            name="char_bigru",
            input_shape=(self.n_words,self.n_characters,self.n_char_rnn_units,self.embedding_output_dim),
            n_units = self.n_char_rnn_units
        )
        self.word_level_settings = BiGruSettings(
            name = "word_bigru",
            input_shape=(self.n_words,self.n_char_rnn_units*2),
            n_units = self.n_word_rnn_units
        )

@dataclass
class OuterModelSettings:
    name: str
    inner_settings: InnerModelSettings
    distance_settings: DistanceSettings
    n_dense_units: int = 128
    p_dropout: float = .5
    loss: tf.losses.Loss = tf.keras.losses.BinaryCrossentropy()
    optimizer : tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(1e-3)
    metrics : list[tf.keras.metrics.Metric] = field(default_factory=list)

@dataclass
class FitSettings:
    batch_size: int=1000
    epochs: int=20
    verbose: int=1
    callbacks: list[tf.keras.callbacks.Callback]=field(default_factory=list)


@dataclass
class ExperimentSettings:
    """The class is responsible for storing one experiment"""
    experiment_name:str
    log_dir:str
    outer_settings: OuterModelSettings = None
    fit_settings: FitSettings = None
    checkpoint:str=None

    def __post_init__(self):
        """
        Set up callbacks
        if the checkpoint is not passed, a new directory will be created for callbaks, 
        otherwise, checkpoints will be added to the existing ones
        """
        if self.checkpoint:
            self.log_dir = self.log_dir + self.checkpoint
        else:
            self.log_dir = self.log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        print(f"logs will be saved to: {self.log_dir}")
        self.checkpoint_path = self.log_dir + "/weights/cp-{epoch:02d}.ckpt"

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir, 
            histogram_freq=1
        )

        self.checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = self.checkpoint_path,
            verboise=1,
            save_weights_only=True,
            save_best_only=False
        )
        #checkpoints_callback.params
        self.fit_settings.callbacks = [self.tensorboard_callback,self.checkpoints_callback]
        print(repr(self))