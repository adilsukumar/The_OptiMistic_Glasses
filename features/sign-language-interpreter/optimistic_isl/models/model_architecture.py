"""Model architectures for Optimistic ISL dynamic/static sign recognition.

Compares three lightweight architectures suitable for Raspberry Pi deployment:
A. mlp_static: Temporal pooling + MLP (baseline; discards temporal order).
B. gru_temporal: Unrolled stacked GRU (preserves order; pure TFLITE_BUILTINS).
C. tcn_temporal: Lightweight 1D Temporal ConvNet (recommended for Pi latency & accuracy).
D. hybrid_cnn_gru: Conv1D temporal stem + unrolled GRU.
"""

from __future__ import annotations

from typing import Dict

import keras
from keras import layers, regularizers


ARCHITECTURE_DESCRIPTIONS: Dict[str, str] = {
    "mlp_static": (
        "Temporal mean-pooled landmark MLP baseline. Fast, but loses stroke/motion "
        "order needed for dynamic ISL signs."
    ),
    "gru_temporal": (
        "Unrolled 2-layer GRU sequence model. Captures sequential dynamics and "
        "exports to pure TFLITE_BUILTINS when unrolled over fixed T=30."
    ),
    "tcn_temporal": (
        "Lightweight 1D Temporal Convolutional Network (Conv1D + SeparableConv1D "
        "+ residual + dual global pooling). Best latency/accuracy tradeoff for "
        "Raspberry Pi 4/5 with zero Flex ops."
    ),
    "hybrid_cnn_gru": (
        "1D Conv stem (2x temporal stride) followed by an unrolled GRU layer."
    ),
}


def build_model(
    architecture: str,
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    dropout_rate: float = 0.30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int | None = None,
) -> keras.Model:
    """Build and compile a Keras model for ISL landmark sequence classification."""
    if architecture not in ARCHITECTURE_DESCRIPTIONS:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from {list(ARCHITECTURE_DESCRIPTIONS.keys())}"
        )

    reg = regularizers.l2(weight_decay) if weight_decay > 0 else None
    inputs = keras.Input(
        shape=(sequence_length, feature_dim),
        batch_size=batch_size,
        name="landmark_sequence",
    )

    if architecture == "mlp_static":
        x = layers.GlobalAveragePooling1D()(inputs)
        x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
        x = layers.Dropout(dropout_rate * 0.7)(x)

    elif architecture == "gru_temporal":
        x = layers.SpatialDropout1D(0.10)(inputs)
        x = layers.GRU(
            64,
            return_sequences=True,
            unroll=True,
            kernel_regularizer=reg,
        )(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.GRU(
            48,
            return_sequences=False,
            unroll=True,
            kernel_regularizer=reg,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)

    elif architecture == "tcn_temporal":
        x = layers.SpatialDropout1D(0.10)(inputs)
        # Stem projection
        x = layers.Conv1D(
            96,
            kernel_size=3,
            padding="same",
            use_bias=False,
            kernel_regularizer=reg,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # Residual depthwise-separable temporal block (dilation=1)
        res = x
        y = layers.SeparableConv1D(
            96,
            kernel_size=3,
            padding="same",
            use_bias=False,
            depthwise_regularizer=reg,
            pointwise_regularizer=reg,
        )(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation("relu")(y)
        y = layers.Dropout(dropout_rate * 0.5)(y)
        y = layers.SeparableConv1D(
            96,
            kernel_size=3,
            padding="same",
            dilation_rate=2,
            use_bias=False,
            depthwise_regularizer=reg,
            pointwise_regularizer=reg,
        )(y)
        y = layers.BatchNormalization()(y)
        x = layers.Add()([res, y])
        x = layers.Activation("relu")(x)

        # Second temporal abstraction stage
        x = layers.SeparableConv1D(
            128,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            depthwise_regularizer=reg,
            pointwise_regularizer=reg,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        avg_pool = layers.GlobalAveragePooling1D()(x)
        max_pool = layers.GlobalMaxPooling1D()(x)
        x = layers.Concatenate()([avg_pool, max_pool])
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(96, activation="relu", kernel_regularizer=reg)(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)

    elif architecture == "hybrid_cnn_gru":
        x = layers.SpatialDropout1D(0.10)(inputs)
        x = layers.Conv1D(
            96,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            kernel_regularizer=reg,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.GRU(
            64,
            return_sequences=False,
            unroll=True,
            kernel_regularizer=reg,
        )(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="class_probs")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"optimistic_{architecture}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
