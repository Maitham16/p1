import csv
import json
import os
import socket
import threading
from kafka import KafkaConsumer
import pandas as pd
import tensorflow as tf
import tempfile
import struct
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SERVER_HOST = 'localhost'
SERVER_PORT = 12345
SERVER_SEND_PORT = 12346
BROKER_ADDRESS = 'localhost:9092'
TOPIC_NAMES = ['node1_server_data', 'node2_server_data', 'node3_server_data', 'node4_server_data']
NUM_NODES = 4
EXPECTED_COLUMNS = 18
filename = "central_server_data.csv"

class LoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info(f"End of Epoch {epoch + 1}. Loss: {logs.get('loss')}, Accuracy: {logs.get('accuracy')}")

def write_data_to_csv(writer, data, columns):
    try:
        writer.writerow([data[col] for col in columns])
    except Exception as e:
        print(f"Failed to write data to CSV: {e}")

# Utility Functions
def save_data_to_csv(data):
    """Save provided data to CSV."""
    columns = [
        "timestamp", "car_id", "model", "current_speed", "battery_capacity",
        "charge", "consumption", "location", "node", "car_status",
        "distance_covered", "battery_life", "distance_to_charging_point",
        "weather", "traffic", "road_gradient", "emergency", "emergency_duration"
    ]

    try:
        file_exists = os.path.isfile('central_server_data.csv')

        with open('central_server_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(columns)  # Write headers if file doesn't exist
            write_data_to_csv(writer, data, columns)
    except Exception as e:
        print(f"Failed to write to CSV: {e}")
        return
    
    print(f"Saved data to CSV: {data}")

def send_large_data(sock, data):
    """Send large data over a socket."""
    data_size = len(data)
    sock.sendall(struct.pack('!I', data_size))
    sock.sendall(data)

def receive_large_data(sock):
    """Receive large data over a socket."""
    data_size = struct.unpack('!I', sock.recv(4))[0]
    received_data = b''
    while len(received_data) < data_size:
        more_data = sock.recv(data_size - len(received_data))
        if not more_data:
            raise ValueError("Received less data than expected!")
        received_data += more_data
    return received_data

def send_global_model_to_node(client_socket, client_address):
    """Send the global model to a node."""
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        with global_model_lock:  # Locking while accessing global_model
            global_model.save(tmp.name, save_format="h5")
            serialized_model = tmp.read()

        logger.info("Sending global model to node (%s)", client_address[0])
        try:
            send_large_data(client_socket, serialized_model)
        except BrokenPipeError:
            print("Client disconnected before data could be sent.")
        client_socket.close()

def process_data(row):
    """Process data and return feature set and label."""
    # Mapping indices to the columns from the CSV
    data = {
        "charge": row[5],
        "distance_to_charging_point": row[12],
        "current_speed": row[3],
        "battery_capacity": row[4],
        "consumption": row[6],
        "distance_covered": row[10],
        "battery_life": row[11],
        "emergency_duration": row[17]
    }

    data['needs_charge'] = 1 if float(data['charge']) <= 50 else 0
    features = [
        data["current_speed"],
        data["battery_capacity"],
        data["charge"],
        data["consumption"],
        data["distance_covered"],
        data["battery_life"],
        data["distance_to_charging_point"],
        data["emergency_duration"]
    ]
    if np.isinf(features[6]):
        features[6] = np.nan
    label = data['needs_charge']
    return features, label

def load_csv_data(filename, num_features=8):
    """Load data from CSV and return features and labels."""
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)

    # Check if data is one-dimensional
    if len(data.shape) == 1:
        data = data.reshape(1, -1)  # Reshape to 2D

    X = data[:, :num_features]
    y = data[:, num_features]

    return X, y

# Model Related Functions
def create_blank_model(input_features=8):
    """Create and return a blank model."""
    model = tf.keras.models.Sequential([
        layers.Dense(12, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(input_features,)),  # L2 regularization
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # L2 regularization
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Global Model Initialization
input_features = 8  
global_model = create_blank_model(input_features)
global_model_lock = threading.Lock()
received_models_lock = threading.Lock() 
received_models = []
received_accuracies = []

def train_global_model():
    """Train the global model."""
    print("train_global_model function is called!") 

    # Load raw data
    raw_data = np.genfromtxt("central_server_data.csv", delimiter=',', skip_header=1)

    # Process data
    processed_data = [process_data(row) for row in raw_data]
    X = np.array([item[0] for item in processed_data])
    y = np.array([item[1] for item in processed_data])

    with global_model_lock:
        global_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Log the start of training
        logger.info("Starting training of the global model...")
        
        global_model.fit(X, y, epochs=100, batch_size=32, verbose=1,
                         callbacks=[LoggingCallback()])
        
        score = global_model.evaluate(X, y, verbose=1)
        
        logger.info("Test loss: %s", score[0])
        logger.info("Test accuracy: %s", score[1])

        # Log the completion of training
        logger.info("Training of the global model completed.")

def aggregate_models_federated_averaging(models, accuracies):
    """Aggregate models using Federated Averaging."""
    # Calculate the weights for each model
    weights = [accuracy / sum(accuracies) for accuracy in accuracies]

    # Get the reference weights from the first model
    reference_weights = models[0].get_weights()
    averaged_weights = []

    for i in range(len(reference_weights)):
        layer_weights_list = np.array([model.get_weights()[i] for model in models])
        averaged_weights.append(np.average(layer_weights_list, axis=0, weights=weights))

    with global_model_lock:
        global_model.set_weights(averaged_weights)
    global_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Kafka and Networking Functions
def handle_client_connection(client_socket, client_address):
    """Handle the client connection for incoming models."""
    global received_models, received_accuracies
    
    print(f"Handling connection from client {client_address[0]}")

    try:
        # Receiving the accuracy and model
        accuracy = float(client_socket.recv(1024).decode())
        logger.info("Received accuracy: %s from client %s", accuracy, client_address[0])
        data = receive_large_data(client_socket)

        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            tmp_file.write(data)
            local_model = tf.keras.models.load_model(tmp_file.name, compile=False)

            with received_models_lock: 
                received_models.append(local_model)
                received_accuracies.append(accuracy)
                all_models_received = len(received_models)

        # Aggregating
        if all_models_received == 4:
            with received_models_lock:
                logger.info("All models received. Aggregating...")
                aggregate_models_federated_averaging(received_models, received_accuracies)
                received_models = []
                received_accuracies = []
                logger.info("Aggregation complete.")

            # Training the aggregated global model
            train_global_model()

        print("About to send READY confirmation to client")
        client_socket.send("READY".encode())
        logger.info("Sent READY confirmation to client")

    except Exception as e:
        logger.error(f"Error handling client {client_address[0]}: {e}")

    finally:
        client_socket.close()

def consume_kafka_messages(topic_names):
    """Consume messages from Kafka topics."""
    print("Starting Kafka consumer...")
    msg = None
    try:
        consumer = KafkaConsumer(
            *topic_names,
            bootstrap_servers='localhost:9092',
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id="central_server"
        )

        for _, msg in enumerate(consumer):
            data = msg.value
            print(f"Processed Kafka message from topic {msg.topic}: {data}")
            save_data_to_csv(data)

    except Exception as e:
        error_msg = f"Kafka consumption error: {e}"
        if msg:
            error_msg += f", Last processed message topic: {msg.topic}"
        print(error_msg)

        
# Main Server Functions
if __name__ == "__main__":

    def kafka_consumer_thread():
            """Start Kafka Consumer."""
            consume_kafka_messages(TOPIC_NAMES)

    def receive_thread():
        """Listen for incoming models from nodes."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((SERVER_HOST, SERVER_PORT))
            s.listen()
            logger.info("Server listening for incoming models on %s:%s", SERVER_HOST, SERVER_PORT)

            while True:
                client_socket, client_address = s.accept()
                logger.info("Accepted connection from %s:%s", client_address[0], client_address[1])
                threading.Thread(target=handle_client_connection, args=(client_socket, client_address)).start()

    def send_thread():
        """Send the aggregated model back to nodes."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as send_socket:
            send_socket.bind((SERVER_HOST, SERVER_SEND_PORT))
            send_socket.listen()
            logger.info("Server ready to send global models on %s:%s", SERVER_HOST, SERVER_SEND_PORT)

            while True:
                client_socket, client_address = send_socket.accept()
                threading.Thread(target=send_global_model_to_node, args=(client_socket, client_address)).start()

    # Start server threads
    threading.Thread(target=kafka_consumer_thread).start()
    threading.Thread(target=receive_thread).start()
    threading.Thread(target=send_thread).start()