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

################### Socket Related Functions ###################
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
        with global_model_lock:
            global_model.save(tmp.name, save_format="h5")
        serialized_model = tmp.read()

        logger.info("Sending global model to node (%s)", client_address[0])
        send_large_data(client_socket, serialized_model)
        
        client_socket.close()

################### Model Related Functions ###################
def create_blank_model():
    """Create a blank model with the same architecture as the local model."""
    model = tf.keras.models.Sequential([
        layers.Dense(12, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(8,)),
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Initialize the global model
global_model = create_blank_model()
global_model_lock = threading.Lock()
received_models_lock = threading.Lock() 
received_models = []
received_accuracies = []

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

################### Server Related Functions ###################
def handle_client_connection(client_socket, client_address):
    """Handle a connection from a node."""
    global received_models, received_accuracies, global_model, all_models_received

    accuracy = float(client_socket.recv(1024).decode())
    logger.info("Received accuracy: %s from client %s", accuracy, client_address[0])

    # Step 1: Receive local model from the node
    data = receive_large_data(client_socket)
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        tmp_file.write(data)
        local_model = tf.keras.models.load_model(tmp_file.name, compile=False)
        local_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        logger.info("Received local model from client")

        with received_models_lock:
            received_models.append(local_model)
            received_accuracies.append(accuracy)
            all_models_received = len(received_models) == NUM_NODES

    if len(received_models) == AGGREGATION_THRESHOLD:
        with received_models_lock:
            logger.info("All models received. Aggregating...")
            aggregate_models_simple_average(received_models)
            received_models = []
            received_accuracies = []
            logger.info("Aggregation complete.")

        # Summary of the aggregated model
        with global_model_lock:
            global_model.summary(print_fn=logger.info)

        # Send the updated global model back to all nodes
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            with global_model_lock:
                global_model.save(tmp.name, save_format="h5")
                tmp.flush()
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

################### Main Function ###################
if __name__ == "__main__":
    ################### Threads ###################
    def receive_thread():
        """Thread to receive models from nodes."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((SERVER_HOST, SERVER_PORT))
            s.listen()
            logger.info("Server listening for incoming models on %s:%s", SERVER_HOST, SERVER_PORT)

            while True:
                client_socket, client_address = s.accept()
                logger.info("Accepted connection from %s:%s", client_address[0], client_address[1])
                threading.Thread(target=handle_client_connection, args=(client_socket, client_address)).start()

    def send_thread():
        """Thread to send global models to nodes."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as send_socket:
            send_socket.bind((SERVER_HOST, SERVER_SEND_PORT))
            send_socket.listen()
            logger.info("Server ready to send global models on %s:%s", SERVER_HOST, SERVER_SEND_PORT)

            while True:
                client_socket, client_address = send_socket.accept()
                threading.Thread(target=send_global_model_to_node, args=(client_socket, client_address)).start()

    # Start the threads
    threading.Thread(target=receive_thread).start()
    threading.Thread(target=send_thread).start()