import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model_path = "./model/modelCNN"
model = load_model(model_path)
print("Model loaded successfully!")

# Load test data
def get_test_data():
    def unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data

    # Path to CIFAR-10 test batch
    test_batch_path = "./cifar-10-batches-py/test_batch"

    # Load test data
    test_data_dict = unpickle(test_batch_path)
    test_data = test_data_dict[b'data']
    test_labels = np.array(test_data_dict[b'labels'])

    # Reshape and normalize the data
    test_data = test_data.reshape(-1, 32, 32, 3) / 255.0

    return test_data, test_labels

# Get the test dataset
test_data, test_labels = get_test_data()
print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

# Make predictions
predictions = model.predict(test_data)

# Convert probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)
print(f"Sample predictions: {predicted_labels[:10]}")

from sklearn.metrics import accuracy_score, classification_report

# Calculate accuracy
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Test Accuracy: {accuracy}")

# Print detailed classification report
print("Classification Report:")
print(classification_report(test_labels, predicted_labels))
