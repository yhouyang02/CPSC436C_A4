import os
import argparse
import pickle
import tensorflow as tf
from tensorflow import keras

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    # Local directories
    parser.add_argument("--train", type=str, default="./train.cnn", help="Path to training data")
    parser.add_argument("--model_dir", type=str, default="./model", help="Path to save the trained model")

    # S3 directories (optional for upload)
    parser.add_argument("--s3_bucket", type=str, help="S3 bucket name")
    parser.add_argument("--s3_model_path", type=str, help="S3 path to save the model")

    return parser.parse_args()

def load_dataset(path):
    """
    Load training dataset from a pickle file.
    """
    with open(path, "rb") as f:
        train_labels, train_data = pickle.load(f)
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

def upload_to_s3(local_path, bucket, s3_path):
    """
    Upload a file to S3.
    """
    s3 = boto3.client("s3")
    print(f"Uploading {local_path} to s3://{bucket}/{s3_path}")
    s3.upload_file(local_path, bucket, s3_path)
    print("Upload complete.")

if __name__ == "__main__":
    args = parse_args()

    # Ensure the model directory exists
    os.makedirs(args.model_dir, exist_ok=True)

    # Load the training dataset
    print("Loading training data...")
    X_train, y_train = load_dataset(args.train)

    # Build the CNN model
    print("Building the CNN model...")
    model = build_model()

    # Train the model
    print("Starting training...")
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1)

    # Save the model locally
    model_path = os.path.join(args.model_dir, "modelCNN")
    print(f"Saving the model to {model_path}...")
    model.save(model_path)