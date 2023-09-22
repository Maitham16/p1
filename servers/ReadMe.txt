# The central server scripts are:
#   - central_server_nn: 		        receives from nodes neural network models, and accuracies. Then aggregate them, 
                                        and after that sending back the global model to nodes.
#   - central_server_nn_withData: 	    receives from nodes neural network models, accuracies, and data. Then aggregate them, 
                                        also train the gloabl model, and after that sending back the global model to nodes.
#   - central_server_lr: 		        receives from nodes linear regression models, accuracies, and data. Then aggregate them, 
                                        and after that sending back the global model to nodes. 
                                        (train the global model is paused currently)
                                        