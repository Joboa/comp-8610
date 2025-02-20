CONFIG = {
    "hyper_params": {
        # Hyperparameters
        "batch_size": 64,
        "num_epochs": 4,
        "learning_rate": 1e-3,
        "optimizer": "Adam", 
        "decay_value": 0
    },
    "net_params": {
        # Network parameters
        "input_size": 28*28,
        "hidden_layers_sizes": [128, 64, 32],
        "output_size": 10,
    },
    "validation": {
        # k-fold cross validation
        "folds": 2,
    },
    "paths": {
        "results": "./results"
    }
}
