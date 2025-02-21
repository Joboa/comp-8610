import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from torch import optim
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from config import CONFIG
from model import MNISTANN

# selected_optimizer = CONFIG["hyper_params"]["optimizer"]


def data_iter(data_loader):
    """
    Return batch of data
    """
    batch_data = iter(data_loader)
    images, labels = next(batch_data)

    return images, labels


def optimizers(model, optimizer_name, learning_rate, decay_value):
    """Returns an optimization type based on the selected optimizer"""
    return getattr(optim, optimizer_name)(model.parameters(),
                                              lr=learning_rate,
                                              weight_decay=decay_value)


def train_model(num_epochs, model, optimizer, loss_fn, train_dataloader):
    """Train model based on the training datasets"""
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            output = model(images)
            loss = loss_fn(output, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if (i+1) % 1000 == 0:
                print(
                    f'Epoch: {epoch+1}/{num_epochs},Step: {i+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}')

        average_loss = epoch_loss / len(train_dataloader)
        accuracy = 100 * correct / total

        train_losses.append(average_loss)
        train_accuracies.append(accuracy)
        print(
            f"Epoch: {epoch + 1} completed, Average Loss: {average_loss:.4f}")

    print("Training completed")
    return train_losses, train_accuracies


def test_model(model, test_loader, criterion):
    """Test model based on the testing datasets"""
    model.eval()
    batch_losses = []
    batch_accuracies = []

    correct = 0
    total = 0

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)

            batch_losses.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)

            batch_accuracy = 100 * batch_correct / batch_total
            batch_accuracies.append(batch_accuracy)

            correct += batch_correct
            total += batch_total

            # For confusion matrix
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    avg_loss = sum(batch_losses) / len(batch_losses)
    avg_accurracy = 100 * correct / total

    return avg_loss, avg_accurracy, batch_losses, batch_accuracies, true_labels, pred_labels


# K-Fold Cross-Validation
def cross_validation(n_splits,
                     num_epochs,
                     decay_value,
                     input_size,
                     hidden_layers_sizes,
                     output_size,
                     learning_rate,
                     train_dataset,
                     batch_size,
                     shuffle_data=True, ):
    """Performs cross validation using sklearn KFold on the training datasets"""

    kfold = KFold(n_splits, shuffle=shuffle_data)
    results = {'train_loss': [],
               'train_acc': [],
               'val_loss': [],
               'val_acc': []}
    # confusion_matrices = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'Fold {fold + 1}')

        # Training and validation subsets
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        print(f"Length of train subset: {len(train_subset)}")

        # Dataloader for Train and validation sets
        trainloader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False)

        # Loss and accuracy for each fold
        fold_train_loss = []
        fold_train_acc = []
        fold_val_loss = []
        fold_val_acc = []

        # For confusion matrix
        # true_labels = []
        # pred_labels = []

        model = MNISTANN(input_size, hidden_layers_sizes, output_size)
        optimizer = optimizers(model, learning_rate, decay_value=decay_value)
        loss_fn = nn.CrossEntropyLoss()

        train_loss, train_acc = train_model(
            num_epochs, model, optimizer, loss_fn, trainloader)
        fold_train_loss.extend(train_loss)
        fold_train_acc.extend(train_acc)

        # Validate the model
        val_loss, val_acc, _, _, true_label, pred_label = test_model(model, valloader, loss_fn)
        fold_val_loss.append(val_loss)
        fold_val_acc.append(val_acc)

        # Labels and predictions for confusion matrix
        # true_labels.extend(true_label)
        # pred_labels.extend(pred_label)

        # Update confusion matrix for the fold
        # cm = confusion_matrix(true_labels, pred_labels)
        # confusion_matrices.append(cm)

        # Update results
        results['train_loss'].append(fold_train_loss[-1])
        results['train_acc'].append(fold_train_acc[-1])
        results['val_loss'].append(fold_val_loss[-1])
        results['val_acc'].append(fold_val_acc[-1])

        # plt.figure(figsize=(8, 6))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(output_size), yticklabels=range(output_size))
        # plt.xlabel("Predicted")
        # plt.ylabel("True")
        # plt.title("Average Confusion Matrix over K-Folds")
        # plt.show()

        # Loss/accuracy for the fold based on the last epoch's values
        print(
            f'Training Loss: {fold_train_loss[-1]:.4f}, Training Accuracy: {fold_train_acc[-1]:.2f}%')
        print(
            f'Validation Loss: {fold_val_loss[-1]:.2f}, Validation Accuracy: {fold_val_acc[-1]:.2f}%')

    # Average results from all folds
    avg_train_loss = sum(results['train_loss']) / len(results['train_loss'])
    avg_train_acc = sum(results['train_acc']) / len(results['train_acc'])
    avg_val_loss = sum(results['val_loss']) / len(results['val_loss'])
    avg_val_acc = sum(results['val_acc']) / len(results['val_acc'])

    print(
        f'Average Training Loss: {avg_train_loss:.4f}, Average Training Accuracy: {avg_train_acc:.2f}%')
    print(
        f'Average Validation Loss: {avg_val_loss:.4f}, Average Validation Accuracy: {avg_val_acc:.2f}%')


    # avg_confusion_matrix = np.mean(confusion_matrices, axis=0)

    return {
        "train_loss": results['train_loss'],
        "train_acc": results['train_acc'],
        "val_loss": results['val_loss'],
        "val_acc": results['val_acc'],
        # "confusion_matrices": confusion_matrices
    }
