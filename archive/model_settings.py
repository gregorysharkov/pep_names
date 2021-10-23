from dataclasses import dataclass, field
import tensorflow as tf

#settings class
@dataclass
class InnerModelSettings:
    """
    Dataclass for storring inner model settings
    """
    input_embedding: int
    n_embedding_dims: int = 64
    n_gru: int = 20
    n_dense: int = 40
    n_units_attention: int = 10

@dataclass
class OuterModelSettings:
    """
    Dataclass for storring outer model settings
    """
    loss: tf.losses.Loss
    optimizer : tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(1e-3)
    metrics : list[tf.keras.metrics.Metric] = field(default_factory=list)


@dataclass
class FitSettings:
    batch_size: int=1000
    epochs: int=20
    verbose: int=1
    callbacks: list[tf.keras.callbacks.Callback]=field(default_factory=list)