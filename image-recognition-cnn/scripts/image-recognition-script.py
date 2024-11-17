import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)

    # Data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # Model directory
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_args()

def load_dataset(path):
    """
    Load training or testing dataset from a pickle file.
    """
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".cnn")]
    if len(files) == 0:
        raise ValueError(f"No .cnn files found in directory: {path}")

    train_labels, train_data = pickle.load(open(files[0], 'rb'))
    return train_data, train_labels

def build_model():
    """
    Build a CNN model using TensorFlow/Keras.
    """
    model = keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def model_fn(model_dir):
    """
    Load the model for inference.
    """
    return tf.keras.models.load_model(os.path.join(model_dir, "modelCNN"))

if __name__ == "__main__":
    args = parse_args()

    print("Loading training data...")
    X_train, y_train = load_dataset(args.train)

    print("Building the CNN model...")
    model = build_model()

    print("Starting training...")
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1)

    print("Saving the model...")
    model.save(os.path.join(args.model_dir, "modelCNN"))
    print("Model saved successfully.")
    