import logging
import numpy as np
import tensorflow as tf
import json
import joblib
from kafka import KafkaConsumer
import csv
import socket
import threading
import time
import matplotlib.pyplot as plt
import struct 
import pandas as pd
import tempfile
from kafka import KafkaProducer

# Initialize Kafka producer for sending data to the central server
central_server_producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Define constants
BATCH_SIZE = 2500
SERVER_HOST = 'localhost'
SERVER_PORT = 12345
SERVER_SEND_PORT = 12346
KAFKA_TOPIC_TO_SERVER = 'node4_server_data' 

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize global variables and locks
total_predictions = 0
correct_predictions = 0
training_batch = []
accumulated_records = []
nn_model = tf.keras.models.load_model('/home/maith/Desktop/p1/models/neural_network_model_node_4.h5')

# Check if the model is compiled
if not nn_model.optimizer:
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load scaler for feature normalization
scaler = joblib.load('/home/maith/Desktop/p1/models/scaler_node_4.pkl')
model_lock = threading.Lock()
prediction_lock = threading.Lock()
accuracy_list = []

def plot_accuracy():
    """Plot the prediction accuracy over time."""
    global total_predictions
    global correct_predictions
    global accuracy_list

    if total_predictions == 0:
        return

    accuracy = correct_predictions / total_predictions * 100
    accuracy_list.append(accuracy)

    plt.clf()
    plt.plot(accuracy_list, '-o')
    plt.xlabel('Time (updates)')
    plt.ylabel('Prediction Accuracy (%)')
    plt.title('Prediction Accuracy Over Time')
    plt.pause(0.01)

def write_data_to_csv(writer, data, columns):
    """Write data to CSV file."""
    try:
        writer.writerow([data[col] for col in columns])
    except Exception as e:
        logging.error(f"Failed to write data to CSV: {e}")

def process_data(data):
    """Process data and return features and label."""
    data['needs_charge'] = 1 if float(data['charge']) <= 50 else 0
    features = [
        float(data["current_speed"]),
        float(data["battery_capacity"]),
        float(data["charge"]),
        float(data["consumption"]),
        float(data["distance_covered"]),
        float(data["battery_life"]),
        float(data["distance_to_charging_point"]),
        float(data["emergency_duration"])
    ]
    if np.isinf(features[6]):
        features[6] = np.nan
    label = data['needs_charge']
    return features, label

def predict_need_charge(model, scaler, features):
    """Predict whether the car needs to be charged."""
    try:
        feature_names = [
            "current_speed", "battery_capacity", "charge", "consumption",
            "distance_covered", "battery_life", "distance_to_charging_point", 
            "emergency_duration"
        ]
        df = pd.DataFrame([features], columns=feature_names)

        features_scaled = scaler.transform(df)
        prediction = model.predict(features_scaled)
        return int(prediction.round())
    
    except Exception as e:
        logging.error(f"Error in prediction process: {e}")
        return None

def retrain_model(batch):
    """Retrain the model on the given batch of data."""
    global nn_model
    X_train = [item[0] for item in batch]
    y_train = [item[1] for item in batch]
    with model_lock:
        nn_model.train_on_batch(X_train, y_train)

def predict_and_update(data):
    """Predict whether the car needs to be charged and update the model."""
    global total_predictions
    global correct_predictions
    global training_batch
    global nn_model

    logging.info("Processing received data...")

    columns = [
        "timestamp", "car_id", "model", "current_speed", "battery_capacity",
        "charge", "consumption", "location", "node", "car_status",
        "distance_covered", "battery_life", "distance_to_charging_point",
        "weather", "traffic", "road_gradient", "emergency", "emergency_duration"
    ]

    try:
        with open('node4_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            write_data_to_csv(writer, data, columns)
    except Exception as e:
        logging.error(f"Failed to write data to CSV: {e}")
        return

    try:
        features, label = process_data(data)
        with model_lock:
            prediction_nn = predict_need_charge(nn_model, scaler, features)

        logging.info(f"Prediction: {prediction_nn}")

        with prediction_lock:
            total_predictions += 1
            if prediction_nn == label:
                correct_predictions += 1
                plot_accuracy()
            training_batch.append((features, label))
            if len(training_batch) == BATCH_SIZE:
                retrain_model(training_batch)
                training_batch = []

    except Exception as e:
        logging.error(f"Error in prediction and update process: {e}")

def send_large_data(sock, data):
    """Send large data over a socket."""
    data_size = len(data)
    sock.sendall(struct.pack("!I", data_size))
    sock.sendall(data)

def receive_large_data(sock):
    """Receive large data over a socket."""
    data_size = struct.unpack("!I", sock.recv(4))[0]
    chunks = []
    bytes_received = 0
    while bytes_received < data_size:
        chunk = sock.recv(min(data_size - bytes_received, 4096))
        if chunk == b'':
            raise RuntimeError("Socket connection broken")
        chunks.append(chunk)
        bytes_received += len(chunk)
    return b''.join(chunks)

# calculate accuracy
node_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
logging.info(f"Node accuracy: {node_accuracy:.2f}%")

def exchange_model_with_server(local_model):
    """Exchange the local model with the server."""
    MAX_RETRIES = 3
    RETRY_WAIT = 5
    
    logging.info("Starting model exchange with the server.")
    
    # Serialize the local model
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        local_model.save(tmp.name, save_format="h5")
        serialized_model = tmp.read()

    for retry in range(MAX_RETRIES):
        try:
            # Calculate node accuracy
            node_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            accuracy_str = str(node_accuracy)

            # Step 1: Node sends its model to the server
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                logging.info(f"Attempting to connect to the server at {SERVER_HOST}:{SERVER_PORT}")
                s.connect((SERVER_HOST, SERVER_PORT))
                logging.info(f"Successfully connected to the server at {SERVER_HOST}:{SERVER_PORT}")

                logging.info("Sending local model's accuracy to the server...")
                s.sendall(accuracy_str.encode())
                time.sleep(0.5)

                logging.info("Sending local model to the server...")
                send_large_data(s, serialized_model)
                logging.info("Local model sent successfully.")

                # Step 2: Receive a confirmation from the server
                confirmation = s.recv(1024)
                if not confirmation.decode() == "READY":
                    raise Exception(f"Unexpected server confirmation: {confirmation.decode()}")
                logging.info("Received READY confirmation from the server.")

            # Step 3: Connect back to the server to receive the updated global model
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(30)
                logging.info(f"Attempting to connect to the server at {SERVER_HOST}:{SERVER_SEND_PORT} for receiving the model.")
                s.connect((SERVER_HOST, SERVER_SEND_PORT))
                logging.info("Connected successfully.")

                logging.info("Waiting to receive the global model from the server...")
                data = receive_large_data(s)

                # Send an acknowledgment after successful receipt
                s.sendall("ACK".encode())

                with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                    tmp_file.write(data)
                    tmp_file.flush()
                    updated_model = tf.keras.models.load_model(tmp_file.name, compile=False)

                updated_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                # Log details about the received model
                logging.info("Received global model.")
                logging.info(f"Model summary: {updated_model.summary()}")
                logging.info(f"Number of layers in model: {len(updated_model.layers)}")
                logging.info(f"Number of trainable parameters: {updated_model.count_params()}")

                return updated_model

        except socket.timeout:
            logging.error("Socket operation timed out. Retrying...")
            time.sleep(RETRY_WAIT * (retry + 1))
            continue

        except Exception as e:
            logging.error(f"Failed to connect or exchange data with the server: {e}. Retrying...")
            time.sleep(RETRY_WAIT * (retry + 1))
            continue

    logging.error("Failed to exchange model with server after maximum retries.")
    return None

def print_model_accuracy():
    """Print the model accuracy."""
    with prediction_lock:
        node_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        logging.info(f"Node accuracy: {node_accuracy:.2f}%")

def periodic_model_exchange():
    """Periodically exchange the local model with the server."""
    global nn_model, correct_predictions, total_predictions
    while True:
        time.sleep(120)  
        try:
            print_model_accuracy()  # Before exchanging models
            updated_model = exchange_model_with_server(nn_model)
            if updated_model is None:
                logging.error("Model is None.")
            else:
                time.sleep(5)
                with model_lock:
                    nn_model = updated_model

                correct_predictions = 0
                total_predictions = 0
            print_model_accuracy()  # After exchanging models
        except Exception as e:
            logging.error(f"Error during model exchange: {e}")

def send_accumulated_data_to_server():
    """Send the accumulated data to the central server."""
    global accumulated_records
    try:
        if accumulated_records:
            for record in accumulated_records:
                central_server_producer.send(KAFKA_TOPIC_TO_SERVER, value=record)
            central_server_producer.flush()
            logging.info(f"Sent {len(accumulated_records)} records to the central server via Kafka.")
            accumulated_records = []
    except Exception as e:
        logging.error(f"Failed to send data to the central server via Kafka: {e}")

def consume_kafka_messages_and_send_to_server(topic_name):
    """Consume messages from a Kafka topic and send them to the central server."""
    try:
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers='localhost:9092',
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='earliest'
        )

        for _, msg in enumerate(consumer):
            data = msg.value
            logging.info(f"Received from {topic_name}: {data}")
            
            # Append the received data to accumulated_records
            accumulated_records.append(data)

            if len(accumulated_records) >= BATCH_SIZE:
                send_accumulated_data_to_server()

    except Exception as e:
        logging.error(f"Kafka consumption error: {e}")

def send_data_to_server(data):
    """Send data to the central server."""
    global accumulated_records
    accumulated_records.append(data)

    if len(accumulated_records) >= BATCH_SIZE:
        send_accumulated_data_to_server()

def consume_kafka_messages(topic_name):
    """Consume messages from a Kafka topic."""
    logging.info("Starting Kafka consumer...")
    try:
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers='localhost:9092',
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='earliest'
        )

        for _, msg in enumerate(consumer):
            data = msg.value
            logging.info(f"Received from {topic_name}: {data}")
            predict_and_update(data)

    except Exception as e:
        logging.error(f"Kafka consumption error: {e}")

# Main execution
if __name__ == "__main__":
    data_send_thread = threading.Thread(target=consume_kafka_messages_and_send_to_server, args=('node4_data',))
    data_send_thread.start()

    model_thread = threading.Thread(target=periodic_model_exchange)
    model_thread.start()
    plt.ion()
    consume_kafka_messages('node4_data')