"""Models package for HAR project."""

from src.models.transformer import (
    TrainablePositionalEmbedding,
    TransformerEncoderBlock,
    build_transformer_classifier,
    compile_transformer_model,
    load_transformer_model,
    predict_transformer,
)
from src.models.bilstm import (
    build_bilstm_model,
    compile_bilstm_model,
    load_bilstm_model,
    predict_bilstm,
)
from src.models.cnn_lstm import (
    build_cnn_lstm_model,
    compile_cnn_lstm_model,
    load_cnn_lstm_model,
    predict_cnn_lstm,
)

__all__ = [
    # Transformer (Dharana - Member 1)
    "TrainablePositionalEmbedding",
    "TransformerEncoderBlock",
    "build_transformer_classifier",
    "compile_transformer_model",
    "load_transformer_model",
    "predict_transformer",
    # BiLSTM (Member 3)
    "build_bilstm_model",
    "compile_bilstm_model",
    "load_bilstm_model",
    "predict_bilstm",
    # CNN-LSTM (Monal - Member 4)
    "build_cnn_lstm_model",
    "compile_cnn_lstm_model",
    "load_cnn_lstm_model",
    "predict_cnn_lstm",
]
