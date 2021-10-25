import tensorflow as tf
from model.v1.model import OuterModel
from model.v1.model_settings import ExperimentSettings

def create_model(experiment_settings:ExperimentSettings) -> OuterModel:
    """
    function creates an OuterModel with provided settings.
    Args:
        inner_settings: an instannce of InnerModelSettings
        outer_settings: an instannce of OuterModelSettings
    """
    model = OuterModel(experiment_settings.outer_settings)
    
    model.compile(
        loss= experiment_settings.outer_settings.loss, 
        optimizer=experiment_settings.outer_settings.optimizer,
        metrics=experiment_settings.outer_settings.metrics,
    )

    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    experiment_settings) -> OuterModel:
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
    model = create_model(experiment_settings)
    model.load_weights(checkpoint)
    return model