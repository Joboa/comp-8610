import torch.nn as nn

from data import mnist_data
from config import CONFIG
from model import MNISTANN
from utils import data_iter, optimizers, train_model, test_model, cross_validation
from visualize import (visualize_training_data,
                       visualize_losses_epochs,
                       visualize_test_data,
                       visualize_cross_validation)

# Hyperparameters
batch_size = CONFIG["hyper_params"]["batch_size"]
learning_rate = CONFIG["hyper_params"]["learning_rate"]
num_epochs = CONFIG["hyper_params"]["num_epochs"]
decay_value = CONFIG["hyper_params"]["decay_value"]
folds = CONFIG["validation"]["folds"]

# Network parameters
input_size = CONFIG["net_params"]["input_size"]
hidden_layers_sizes = CONFIG["net_params"]["hidden_layers_sizes"]
output_size = CONFIG["net_params"]["output_size"]


train_dataset, test_dataset, train_dataloader, test_dataloader = mnist_data(
    batch_size)

############################### Visualize training data  ############################
# images, labels = data_iter(train_dataloader)
# visualize_training_data(images,
#                         labels, 2, 4,
#                         fig_name="visualize_train")


################################## Train model  ######################################
# model = MNISTANN(input_size, hidden_layers_sizes, output_size)
# optimizer = optimizers(model=model, learning_rate=learning_rate, decay_value=0)
# loss_fn = nn.CrossEntropyLoss()
# train_losses, train_accuracies = train_model(
#     num_epochs, model, optimizer, loss_fn, train_dataloader)


####################### Train and visualize loss vs epochs plot  #######################
# visualize_losses_epochs(train_losses, fig_name="losses_epochs")


################################## Visualize testing data  ##############################
# images, labels = data_iter(test_dataloader)
# classes = [str(i) for i in range(10)]

# visualize_test_data(images, labels, model,
#                     classes, n=2, m=5,
#                     fig_name="visualize_test1")

##################### Train and evaluate mode using cross-validation ####################
# optimizer = optimizer(model=model, learning_rate=learning_rate, decay_value=0)

results = cross_validation(n_splits=folds,
                           num_epochs=num_epochs,
                           decay_value=decay_value,
                           input_size=input_size,
                           hidden_layers_sizes=hidden_layers_sizes,
                           output_size=output_size,
                           learning_rate=learning_rate,
                           train_dataset=train_dataset,
                           batch_size=batch_size,
                           shuffle_data=True, )

train_loss = results['train_loss']
train_acc = results['train_acc']
val_loss = results['val_loss']
val_acc = results['val_acc']

# visualize_cross_validation(train_loss, val_loss, train_acc, val_acc)

if __name__ == "__main__":
    # Visualize training data
    # visualize_training_data(images, labels, 2, 4, fig_name="visualize_train")

    # Visualize losses vs epochs plot
    # visualize_losses_epochs(train_losses, fig_name="losses_epochs")

    # Visualize testing data
    # visualize_test_data(images, labels, model, classes,
    #                     n=2, m=5, fig_name="visualize_test1")

    # Visualize cross-validation
    visualize_cross_validation(train_loss, val_loss, train_acc, val_acc)
