from dataclasses import dataclass

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
