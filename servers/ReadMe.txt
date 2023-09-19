The central server scripts are:
- central_server_nn : 		receives from nodes neural network models, and accuracies. Then aggregate them, and sending back the global model to nodes.
- central_server_nn_withData: 	receives from nodes neural network models, accuracies, and data. Then aggregate them, also train the gloabl model, and after that sending back the global model to nodes.
- central_server_ir: 		receives from nodes linear regression models, accuracies, and data. Then aggregate them, also train the gloabl model (this feature is commented for now), and after that sending back the global model to nodes.

