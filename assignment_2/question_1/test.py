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
weight_decays = CONFIG["hyper_params"]["decay_value"]

input_size = CONFIG["net_params"]["input_size"]
hidden_layers_sizes = CONFIG["net_params"]["hidden_layers_sizes"]
output_size = CONFIG["net_params"]["output_size"]

results_dir = CONFIG["paths"]["results"]

# Batch size and learning rate
# selected_batch_size = batch_sizes[0]  # Batch size 16 ##change
# # print(selected_batch_size)
# selected_learning_rate = learning_rates[2]  # Learning rate 0.01  ##change
# selected_hidden_layer = hidden_layers_sizes[0] ## change

# Load dataset
train_dataset, test_dataset, train_dataloader, test_dataloader = mnist_data(
    batch_sizes)

####################################### With different optimizers ################################
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
# kfold = KFold(CONFIG["validation"]["folds"], shuffle=True)
# results_dir = CONFIG["paths"]["results"]
# os.makedirs(results_dir, exist_ok=True)

# for hidden_layers in hidden_layers_sizes:
#     print(f"Training model with architecture: {hidden_layers}")
#     fold_results = {
#         "train_loss": [],
#         "train_acc": [],
#         "val_loss": [],
#         "val_acc": [],
#     }

#     for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
#         print(f"Training fold {fold+1}")

#         # Fold data
#         train_subset = Subset(train_dataset, train_idx)
#         val_subset = Subset(train_dataset, val_idx)
#         train_loader = DataLoader(train_subset, batch_size=selected_batch_size, shuffle=True)
#         val_loader = DataLoader(val_subset, batch_size=selected_batch_size, shuffle=False)

#         # Model
#         model = MNISTANN(input_size, hidden_layers, output_size)

#         # Training and validation for each optimizer
#         fold_train_loss = []
#         fold_train_acc = []
#         fold_val_loss = []
#         fold_val_acc = []

#         for optimizer_name in optimizers_list:
#             print(f"Training with {optimizer_name}")
#             optimizer = optimizers(model, optimizer_name, selected_learning_rate, CONFIG["hyper_params"]["decay_value"])

#             # Train the model
#             train_loss, train_acc = train_model(num_epochs, model, optimizer, torch.nn.CrossEntropyLoss(), train_loader)
#             fold_train_loss.append(train_loss[-1])
#             fold_train_acc.append(train_acc[-1])

#             # Validate the model
#             val_loss, val_acc, _, _, _, _ = test_model(model, val_loader, torch.nn.CrossEntropyLoss())
#             fold_val_loss.append(val_loss)
#             fold_val_acc.append(val_acc)

#         # Store results for this fold
#         fold_results["train_loss"].append(fold_train_loss)
#         fold_results["train_acc"].append(fold_train_acc)
#         fold_results["val_loss"].append(fold_val_loss)
#         fold_results["val_acc"].append(fold_val_acc)

#     # Save results
#     with open(os.path.join(results_dir, f"results_arch_{hidden_layers}.txt"), "w") as file:
#         for key, value in fold_results.items():
#             file.write(f"{key}:\n{np.array(value)}\n\n")

####################################### SGD with and without weight decay ################################
#  Train and plot for each optimizer
# results = {}
# for decay in weight_decays:
#     print(f"Training with decay value of: {decay}")

#     # Model and optimizer
#     model = MNISTANN(input_size, hidden_layers_sizes, output_size)
#     optimizer = optimizers(model, optimizers_list, learning_rates, decay_value=decay)

#     # DataLoader
#     train_loader = train_dataloader
#     loss_fn = torch.nn.CrossEntropyLoss()

#     # Train the model
#     train_losses, train_accuracies = train_model(num_epochs, model, optimizer, loss_fn, train_loader)

#     results[decay] = {
#         "train_losses": train_losses,
#         "train_accuracies": train_accuracies
#     }

#     # Visualize predictions on test data
#     images, labels = next(iter(test_dataloader))
#     classes = [str(i) for i in range(10)]

#     # al_loss, val_acc, _, _, true_label, pred_label = test_model(model, valloader, loss_fn)

#     def visualize_test_data(images, labels, model, classes, n, m, fig_name="visualize_test"):
#         """Visualize test datasets for model evaluation"""

#         model.eval()
#         with torch.no_grad():
#             outputs = model(images)
#             _, preds = torch.max(outputs, 1)

#         _, axes = plt.subplots(n, m, figsize=(10, 5))
#         axes = axes.flatten()
#         N = n * m
#         for i in range(N):
#             img = images[i].squeeze()
#             axes[i].imshow(img.cpu().numpy(), cmap='gray')
#             axes[i].set_title(
#                 f'Pred: {classes[preds[i].item()]}\nTrue: {classes[labels[i].item()]}')
#             axes[i].axis('off')

#         plt.tight_layout()
#         plt.savefig(os.path.join(results_dir, f"{decay}_{fig_name}.png"))
#         plt.show()

#     visualize_test_data(images, labels, model, classes, 2, 5, f"test_predictions_{decay}")


# Plot loss vs epoch for each decay value
# plt.figure(figsize=(10, 6))
# for decay, data in results.items():
#     plt.plot(data["train_losses"], label=decay)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title(f'Training Loss vs Epoch (with Regularization)')
# plt.legend()
# plt.savefig(f'./results/loss_vs_epoch_{decay}.png')
# plt.show()


# Cross-validation
kfold = KFold(CONFIG["validation"]["folds"], shuffle=True)
results_dir = CONFIG["paths"]["results"]
os.makedirs(results_dir, exist_ok=True)

for decay_value in weight_decays:
    print(f"Training model with decay value of: {decay_value}")
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
        train_loader = DataLoader(train_subset, batch_size=batch_sizes, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_sizes, shuffle=False)

        # Setup model and optimizer
        model = MNISTANN(input_size, hidden_layers_sizes, output_size)
        optimizer = optimizers(model, optimizers_list, learning_rates, decay_value)

        # Training and validation for each decay value
        fold_train_loss = []
        fold_train_acc = []
        fold_val_loss = []
        fold_val_acc = []

        train_loss, train_acc = train_model(num_epochs, model, optimizer, torch.nn.CrossEntropyLoss(), train_loader)
        fold_train_loss.append(train_loss[-1])
        fold_train_acc.append(train_acc[-1])

        val_loss, val_acc, _, _, _, _ = test_model(model, val_loader, torch.nn.CrossEntropyLoss())
        fold_val_loss.append(val_loss)
        fold_val_acc.append(val_acc)

        # Store results for each fold
        fold_results["train_loss"].append(fold_train_loss)
        fold_results["train_acc"].append(fold_train_acc)
        fold_results["val_loss"].append(fold_val_loss)
        fold_results["val_acc"].append(fold_val_acc)

    # Save results
    with open(os.path.join(results_dir, f"results_arch_{decay_value}.txt"), "w") as file:
        for key, value in fold_results.items():
            file.write(f"{key}:\n{np.array(value)}\n\n")