import tensorflow as tf
from model_settings import (
    InnerModelSettings, 
    OuterModelSettings, 
    FitSettings
)

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