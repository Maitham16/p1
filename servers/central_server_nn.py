import socket
import threading
import tensorflow as tf
import tempfile
import struct
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
SERVER_HOST = 'localhost'
SERVER_PORT = 12345
SERVER_SEND_PORT = 12346
NUM_NODES = 4
AGGREGATION_THRESHOLD = 4

def create_blank_model():
    model = tf.keras.models.Sequential([
        layers.Dense(12, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(8,)),  # L2 regularization
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # L2 regularization
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Initialize the global model
global_model = create_blank_model()
global_model_lock = threading.Lock()
received_models_lock = threading.Lock() 
received_models = []
received_accuracies = []

# The utility functions from your node script:
def send_large_data(sock, data):
    data_size = len(data)
    sock.sendall(struct.pack('!I', data_size))
    sock.sendall(data)

# Function to receive large data from a socket
def receive_large_data(sock):
    data_size = struct.unpack('!I', sock.recv(4))[0]
    received_data = b''
    while len(received_data) < data_size:
        more_data = sock.recv(data_size - len(received_data))
        if not more_data:
            raise ValueError("Received less data than expected!")
        received_data += more_data
    return received_data

# Function to aggregate the local models into the global model
def aggregate_models_simple_average(models):
    """Aggregate weights from multiple models using a simple average."""
    reference_weights = models[0].get_weights()
    averaged_weights = []

    for i in range(len(reference_weights)):
        layer_weights_list = np.array([model.get_weights()[i] for model in models])
        averaged_weights.append(np.mean(layer_weights_list, axis=0))

    with global_model_lock:
        global_model.set_weights(averaged_weights)
    global_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Connection handler for each node
def handle_client_connection(client_socket, client_address):
    global received_models, received_accuracies

    accuracy = float(client_socket.recv(1024).decode())
    logger.info("Received accuracy: %s from client %s", accuracy, client_address[0])

    # Step 1: Receive local model from the node
    data = receive_large_data(client_socket)
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        tmp_file.write(data)
        local_model = tf.keras.models.load_model(tmp_file.name, compile=False)
        local_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        logger.info("Received local model from client")

        with received_models_lock:  # Locking while updating received_models and received_accuracies
            received_models.append(local_model)
            received_accuracies.append(accuracy)  # Store the accuracy
            all_models_received = len(received_models) == NUM_NODES

    if len(received_models) == AGGREGATION_THRESHOLD:
        with received_models_lock:  # Locking while accessing received_models
            logger.info("All models received. Aggregating...")
            aggregate_models_simple_average(received_models)
            received_models = []
            received_accuracies = []
            logger.info("Aggregation complete.")

        # summary of the aggregated model
        with global_model_lock:
            global_model.summary(print_fn=logger.info)

        # Now send the updated global model back to all nodes
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            with global_model_lock:  # Locking while accessing global_model
                global_model.save(tmp.name, save_format="h5")
            serialized_model = tmp.read()
            
            logger.info("Sending aggregated global model to node")
            send_large_data(client_socket, serialized_model)
    
    # Send a READY confirmation back to the node
    try:
        client_socket.send("READY".encode())
        logger.info("Sent READY confirmation to client")
    except ConnectionResetError:
        logger.error("Connection was reset by client.")
    finally:
        client_socket.close()

def send_global_model_to_node(client_socket, client_address):
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        with global_model_lock:  # Locking while accessing global_model
            global_model.save(tmp.name, save_format="h5")
        serialized_model = tmp.read()

        logger.info("Sending global model to node (%s)", client_address[0])
        send_large_data(client_socket, serialized_model)
        
        # Close the connection after sending
        client_socket.close()

# Main function to start the server...
if __name__ == "__main__":
    # This thread will listen for incoming models from nodes.
    def receive_thread():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((SERVER_HOST, SERVER_PORT))
            s.listen()
            logger.info("Server listening for incoming models on %s:%s", SERVER_HOST, SERVER_PORT)

            while True:
                client_socket, client_address = s.accept()
                logger.info("Accepted connection from %s:%s", client_address[0], client_address[1])
                threading.Thread(target=handle_client_connection, args=(client_socket, client_address)).start()

    # This thread will send the aggregated model back to nodes.
    def send_thread():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as send_socket:
            send_socket.bind((SERVER_HOST, SERVER_SEND_PORT))
            send_socket.listen()
            logger.info("Server ready to send global models on %s:%s", SERVER_HOST, SERVER_SEND_PORT)

            while True:
                client_socket, client_address = send_socket.accept()
                threading.Thread(target=send_global_model_to_node, args=(client_socket, client_address)).start()

    # Start both threads.
    threading.Thread(target=receive_thread).start()
    threading.Thread(target=send_thread).start()