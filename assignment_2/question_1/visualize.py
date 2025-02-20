import torch
import matplotlib.pyplot as plt

from config import CONFIG

results = CONFIG["paths"]["results"]


def visualize_training_data(img, labels, n: int, m: int, fig_name="visualize_train"):
    """
    Visualize sample of training data

    Args:
    img: Sample images for visualization
    labels: Target labels for the images
    N: Number of samples to visualize
    n: rows for the subplot
    m: columns for the subplot
    fig_name: Name of the plot
    """
    _, axes = plt.subplots(n, m, figsize=(12, 4))
    axes = axes.flatten()
    N = n * m
    for i in range(N):
        ax = axes[i]
        ax.imshow(img[i].numpy().squeeze(), cmap="gray")
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")
    plt.savefig(f"{results}/{fig_name}.png")
    plt.show()


def visualize_losses_epochs(losses, fig_name="losses_epochs"):
    """Visualize losses vs epochs on training datasets"""
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.savefig(f"{results}/{fig_name}.png")
    plt.show()


def visualize_test_data(images, labels, model, classes, n, m, fig_name="visualize_test"):
    """Visualiize test datasets for model evaluation"""

    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    _, axes = plt.subplots(n, m, figsize=(10, 5))
    axes = axes.flatten()
    N = n * m
    for i in range(N):
        img = images[i].squeeze()
        axes[i].imshow(img.cpu().numpy(), cmap='gray')
        axes[i].set_title(
            f'Pred: {classes[preds[i].item()]}\nTrue: {classes[labels[i].item()]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{results}/{fig_name}.png")
    plt.show()


def visualize_cross_validation(train_loss, val_loss, train_acc, val_acc):
    plt.figure(figsize=(10, 4))

    folds = range(1, len(train_loss) + 1)

    plt.plot(folds, train_loss, label=f'Train loss', marker="o")
    plt.plot(folds, val_loss, label=f'Validation loss',
             linestyle='dashed',  marker="o")

    plt.xlabel('Number of Folds')
    plt.ylabel('Loss')
    plt.title('Training loss and Validation loss per fold')
    plt.xticks(folds)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(folds, train_acc, label=f'Train accuracy', marker="o")
    plt.plot(folds, val_acc, label=f'Validation accuracy',
             linestyle='dashed',  marker="o")

    plt.xlabel('Number of Folds')
    plt.ylabel('Loss')
    plt.title('Training accuracy vs Validation accuracy per fold')
    plt.xticks(folds)
    plt.legend()
    plt.show()
