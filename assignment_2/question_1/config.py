CONFIG = {
    "hyper_params": {
        # Hyperparameters
        "batch_size": [16, 32, 64],
        "num_epochs": 2,
        "learning_rate": [0.1, 0.01, 0.001],
        "optimizer": ["SGD", "RMSprop", "Adam"], 
        "decay_value": 0
    },
    "net_params": {
        # Network parameters
        "input_size": 28*28,
        "hidden_layers_sizes": [[32]],
        "output_size": 10,
    },
    "validation": {
        # k-fold cross validation
        "folds": 5,
    },
    "paths": {
        "results": "./test_results"
    }
}
