import tensorflow as tf
from model.v1.model_settings import ExperimentSettings, OuterModelSettings, InnerModelSettings, DistanceSettings, FitSettings

zero_to_one_outer_model_settings = OuterModelSettings(
    name="two_level_rnn_with_attention",
    inner_settings=InnerModelSettings(
        name="inner_model",
        n_words=10,
        n_characters=10,
        embedding_input_dim=162,
        embedding_output_dim=256,
        n_char_rnn_units=10,
        n_word_rnn_units=20,
    ),
    distance_settings = DistanceSettings("distance","zero_to_one"),
    n_dense_units=1,
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adadelta(.1),
    metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision")]
)

abs_outer_model_settings = OuterModelSettings(
    name="two_level_rnn_with_attention",
    inner_settings=InnerModelSettings(
        name="inner_model",
        n_words=10,
        n_characters=10,
        embedding_input_dim=162,
        embedding_output_dim=256,
        n_char_rnn_units=20,
        n_word_rnn_units=40,
    ),
    distance_settings = DistanceSettings("distance","abs"),
    n_dense_units=80,
    p_dropout = .7,
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adadelta(),
    metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision")]
)

fit_settings = FitSettings(
    batch_size=125,
    epochs=5,
    verbose=1,
)

BASE_EXPERIMENT = ExperimentSettings(
        experiment_name="2_level_rnn",
        log_dir="logs\\baseline\\base\\",
        outer_settings=zero_to_one_outer_model_settings,
        fit_settings=fit_settings,
        checkpoint=None
)

ABS_EXPERIMENT = ExperimentSettings(
        experiment_name="2_level_rnn",
        log_dir="logs\\baseline\\adadelta\\",
        outer_settings=abs_outer_model_settings,
        fit_settings=fit_settings,
        checkpoint=None
)