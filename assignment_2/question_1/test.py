import os
import matplotlib.pyplot as plt
import torch
import numpy as np 
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import Subset

from utils import train_model, test_model, optimizers
from model import MNISTANN
from config import CONFIG
from data import mnist_data


# Configuration settings
batch_sizes = CONFIG["hyper_params"]["batch_size"]
learning_rates = CONFIG["hyper_params"]["learning_rate"]
optimizers_list = CONFIG["hyper_params"]["optimizer"]
num_epochs = CONFIG["hyper_params"]["num_epochs"]
input_size = CONFIG["net_params"]["input_size"]
hidden_layers_sizes = CONFIG["net_params"]["hidden_layers_sizes"]
output_size = CONFIG["net_params"]["output_size"]

# Batch size and learning rate
selected_batch_size = batch_sizes[0]  # Batch size 16 ##change
# print(selected_batch_size)
selected_learning_rate = learning_rates[2]  # Learning rate 0.01  ##change
selected_hidden_layer = hidden_layers_sizes[0] ## change

# Load dataset
train_dataset, test_dataset, train_dataloader, test_dataloader = mnist_data(
    selected_batch_size)

# Train and plot for each optimizer
# results = {}
# for optimizer_name in optimizers_list:
#     print(f"Training with optimizer: {optimizer_name}")

#     # Model and optimizer
#     model = MNISTANN(input_size, selected_hidden_layer, output_size)
#     optimizer = optimizers(model, optimizer_name, selected_learning_rate, CONFIG["hyper_params"]["decay_value"])

#     # DataLoader
#     train_loader = train_dataloader
#     loss_fn = torch.nn.CrossEntropyLoss()

#     # Train the model
#     train_losses, train_accuracies = train_model(num_epochs, model, optimizer, loss_fn, train_loader)

#     results[optimizer_name] = {
#         "train_losses": train_losses,
#         "train_accuracies": train_accuracies
#     }

# # Plot loss vs epoch for each optimizer
# plt.figure(figsize=(10, 6))
# for optimizer_name, data in results.items():
#     plt.plot(data["train_losses"], label=optimizer_name)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title(f'Training Loss vs Epoch (Batch Size: {selected_batch_size}, LR: {selected_learning_rate})')
# plt.legend()
# plt.savefig(f'./results/loss_vs_epoch_batch_{selected_batch_size}_lr_{selected_learning_rate}.png')
# plt.show()

# Cross-validation
kfold = KFold(CONFIG["validation"]["folds"], shuffle=True)
results_dir = CONFIG["paths"]["results"]
os.makedirs(results_dir, exist_ok=True)

for hidden_layers in hidden_layers_sizes:
    print(f"Training model with architecture: {hidden_layers}")
    fold_results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"Training fold {fold+1}")

        # Fold data
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=selected_batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=selected_batch_size, shuffle=False)

        # Model
        model = MNISTANN(input_size, hidden_layers, output_size)

        # Training and validation for each optimizer
        fold_train_loss = []
        fold_train_acc = []
        fold_val_loss = []
        fold_val_acc = []

        for optimizer_name in optimizers_list:
            print(f"Training with {optimizer_name}")
            optimizer = optimizers(model, optimizer_name, selected_learning_rate, CONFIG["hyper_params"]["decay_value"])

            # Train the model
            train_loss, train_acc = train_model(num_epochs, model, optimizer, torch.nn.CrossEntropyLoss(), train_loader)
            fold_train_loss.append(train_loss[-1])
            fold_train_acc.append(train_acc[-1])

            # Validate the model
            val_loss, val_acc, _, _, _, _ = test_model(model, val_loader, torch.nn.CrossEntropyLoss())
            fold_val_loss.append(val_loss)
            fold_val_acc.append(val_acc)

        # Store results for this fold
        fold_results["train_loss"].append(fold_train_loss)
        fold_results["train_acc"].append(fold_train_acc)
        fold_results["val_loss"].append(fold_val_loss)
        fold_results["val_acc"].append(fold_val_acc)

    # Save results
    with open(os.path.join(results_dir, f"results_arch_{hidden_layers}.txt"), "w") as file:
        for key, value in fold_results.items():
            file.write(f"{key}:\n{np.array(value)}\n\n")
