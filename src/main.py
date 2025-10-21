### ~~~ GLOBAL IMPORTS ~~~ ###
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from typing import TypeAlias, Tuple

### ~~~ LOCAL IMPORTS ~~~ ###
# No local imports for now

### ~~~ CUSTOM TYPES ~~~ ###
tensor_t: TypeAlias = np.ndarray

### ~~~ STATE DEFINITIONS ~~~ ###
DATA_PATH = "dbs/cooked/data.npz"
MODEL_CHECKPOINT_PATH = "models/best_model.keras"
L2_REG = 0.01
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
EPOCHS = 1080
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2

### ~~~ FUNCTION DEFINITIONS ~~~ ###


def load_data(path: str) -> Tuple[tensor_t, tensor_t, tensor_t, tensor_t]:
    """
    Loads training and testing data from a .npz file.

    Args:
        path: The path to the .npz file.

    Returns:
        A tuple containing training features, training labels,
        testing features, and testing labels.
    """
    with np.load(path) as data:
        x_train = data["X_train"]
        y_train = data["y_train"]
        x_test = data["X_test"]
        y_test = data["y_test"]
    return x_train, y_train, x_test, y_test


def build_model_base(
    input_shape: Tuple[int, ...], l2_reg: float, dropout_rate: float
) -> Model:
    """
    Builds a simple feed-forward neural network.
    Args:
        input_shape: The shape of the input data.
        l2_reg: The L2 regularization factor.
        dropout_rate: The dropout rate.

    Returns:
        A compiled Keras model.
    """
    inputs = Input(shape=input_shape)
    x = Dense(128, activation="relu", kernel_regularizer=l2(l2_reg))(inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model: Model = Model(inputs=inputs, outputs=outputs)

    return model


def main() -> None:
    """
    Main script execution function.
    """
    # Load data
    x_train, y_train, x_test, y_test = load_data(DATA_PATH)

    # Build model
    model = build_model_base(x_train.shape[1:], L2_REG, DROPOUT_RATE)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        MODEL_CHECKPOINT_PATH, save_best_only=True, monitor="val_loss"
    )

    # Train model
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping, model_checkpoint],
    )

    # Evaluate model
    loss, accuracy = model.evaluate(x_test, y_test)  # type: ignore
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


### ~~~ SCRIPT EXECUTION ~~~ ###
if __name__ == "__main__":
    main()
