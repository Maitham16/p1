import csv
import json
import os
import socket
import threading
import joblib
from kafka import KafkaConsumer
from sklearn.linear_model import LinearRegression
import tempfile
import struct
import numpy as np
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
is_model_aggregated = False

def write_data_to_csv(writer, data, columns):
    try:
        writer.writerow([data[col] for col in columns])
    except Exception as e:
        logging.error(f"Error writing data to CSV: {e}")

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
        logging.error(f"Error writing data to CSV: {e}")
        return
    
    logging.info("Data written to CSV successfully.")

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
    global is_model_aggregated
    if not is_model_aggregated:
        logger.info("Global model not aggregated yet. Waiting...")
        return
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pkl") as tmp:
        with global_model_lock:  # Locking while accessing global_model
            joblib.dump(global_model, tmp.name)
            serialized_model = tmp.read()
            logger.info("Size of serialized model being sent: %s bytes", len(serialized_model))

        logger.info("Sending global model to node (%s)", client_address[0])
        try:
            send_large_data(client_socket, serialized_model)
            
            # Waiting for an acknowledgment from the client
            acknowledgment = client_socket.recv(1024).decode()
            if acknowledgment == "ACK":
                logger.info("Node (%s) acknowledged receipt of the global model.", client_address[0])
            else:
                logger.warning("Unexpected acknowledgment from node (%s): %s", client_address[0], acknowledgment)
                
        except BrokenPipeError:
            logger.error("Broken pipe error while sending global model to node (%s)", client_address[0])
        except Exception as e:
            logger.error(f"Failed to receive acknowledgment from node ({client_address[0]}): {e}")
            
        finally:
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
    # Replace infinity values with NaN
    if np.isinf(features[6]):
        features[6] = np.nan
    # Replace NaN values with mean values
    features = np.nan_to_num(features, nan=np.nanmean(features))
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
def create_blank_model():
    """Create and return a blank linear regression model."""
    model = LinearRegression()
    return model

# Global Model Initialization
global_model = create_blank_model()
global_model_lock = threading.Lock()
received_models_lock = threading.Lock() 
received_models = []
received_accuracies = []

def train_global_model():
    """Train the global model."""
    logging.info("Training the global model function is called...")

    # Load raw data
    raw_data = np.genfromtxt("central_server_data.csv", delimiter=',', skip_header=1)

    # Process data
    processed_data = [process_data(row) for row in raw_data]
    X = np.array([item[0] for item in processed_data])
    y = np.array([item[1] for item in processed_data])

    with global_model_lock:
        # Log the start of training
        logger.info("Starting training of the global model...")
        
        global_model.fit(X, y)
        
        accuracy = global_model.score(X, y)
        
        logger.info("Test accuracy: %s", accuracy)

        # Log the completion of training
        logger.info("Training of the global model completed.")

def aggregate_models_federated_averaging(models, accuracies):
    """Aggregate models using Federated Averaging."""
    global is_model_aggregated
    logging.info("Aggregating models using Federated Averaging...")

    total_coef = np.zeros_like(models[0].coef_)
    total_intercept = 0

    # Sum all model weights
    for model in models:
        total_coef += model.coef_
        total_intercept += model.intercept_

    # Average the weights
    avg_coef = total_coef / len(models)
    avg_intercept = total_intercept / len(models)

    with global_model_lock:
        global_model.coef_ = avg_coef
        global_model.intercept_ = avg_intercept

        logging.info("Aggregation complete.")
        is_model_aggregated = True

# Kafka and Networking Functions
def handle_client_connection(client_socket, client_address):
    """Handle the client connection for incoming models."""
    global received_models, received_accuracies, is_model_aggregated, all_models_received
    
    logging.info("Handling client connection from %s:%s", client_address[0], client_address[1])

    try:
        # Receiving the accuracy and model
        accuracy = float(client_socket.recv(1024).decode('utf-8'))
        logger.info("Received accuracy: %s from client %s", accuracy, client_address[0])
        data = receive_large_data(client_socket)

        with tempfile.NamedTemporaryFile(delete=True, suffix=".pkl") as tmp_file:
            tmp_file.write(data)
            tmp_file.flush()
            local_model = joblib.load(tmp_file.name)

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
                is_model_aggregated = True
                logger.info("Aggregation complete.")

            # Training the aggregated global model
            # train_global_model()

        logging.info("Sending READY confirmation to client %s", client_address[0])
        client_socket.send("READY".encode())
        logger.info("Sent READY confirmation to client")

    except Exception as e:
        logger.error(f"Error handling client {client_address[0]}: {e}", exc_info=True)

    finally:
        client_socket.close()

def consume_kafka_messages(topic_names):
    """Consume messages from Kafka topics."""
    logging.info("Consuming Kafka messages...")
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
            logging.info(f"Processed Kafka message from topic {msg.topic}: {data}")
            save_data_to_csv(data)

    except Exception as e:
        error_msg = f"Kafka consumption error: {e}"
        if msg:
            error_msg += f", Last processed message topic: {msg.topic}"
        logging.error(error_msg)

        
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