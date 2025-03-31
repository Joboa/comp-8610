import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from gnn import GraphConvAutoencoder, custom_collate, UnstructuredDataset, parse_data


def load_model_and_data(model_path, test_loader, device):
    """
    Load the trained model and prepare test data.

    Args:
        model_path (str): Path to the saved model checkpoint
        test_loader (DataLoader): DataLoader for test set
        device (torch.device): Computing device

    Returns:
        tuple: (x_coords, y_coords, actual_data, predicted_data, error_data)
    """
    num_features = 1
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = GraphConvAutoencoder(in_channels=num_features)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    actual_data, predicted_data, error_data = [], [], []
    x_coords, y_coords = [], []

    folder_path = "./data/datasets/CSW"

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            reconstructed_x = model(batch)
            actual = batch.x.cpu().numpy().flatten()
            predicted = reconstructed_x.cpu().numpy().flatten()
            error = np.abs(actual - predicted)
            actual_data = [actual]
            predicted_data = [predicted]
            error_data = [error]
            break 

    # Coordinates for unstructured data
    dat_files = [f for f in os.listdir(folder_path) if f.endswith(".dat")]
    x, y, _ = parse_data(os.path.join(folder_path, dat_files[0]))

    actual_data = np.concatenate(actual_data)
    predicted_data = np.concatenate(predicted_data)
    error_data = np.concatenate(error_data)
    # actual_data = np.concatenate(actual_data)[:len(x)]
    # predicted_data = np.concatenate(predicted_data)[:len(x)]
    # error_data = np.concatenate(error_data)[:len(x)]

    if len(x) != len(actual_data):
        print(
            f"Warning: Coordinate length ({len(x)}) does not match data length ({len(actual_data)})")

    return x, y, actual_data, predicted_data, error_data

def plot_velocity_contours(x_coords, y_coords, actual_data, predicted_data, error_data, output_dir):
    """
    Create contour plots for actual velocity, predicted velocity, and error for unstructured data,
    normalized to a range of -1 to 1 for easier comparison.

    Args:
        x_coords (np.ndarray): X-coordinates of data points
        y_coords (np.ndarray): Y-coordinates of data points
        actual_data (np.ndarray): Actual velocity values
        predicted_data (np.ndarray): Predicted velocity values
        error_data (np.ndarray): Absolute error between actual and predicted
        output_dir (str): Directory to save output plots
    """
    # Normalize data to range [-1, 1]
    def normalize(data, range_min=-1, range_max=1):
        data_min = data.min()
        data_max = data.max()
        return (data - data_min) / (data_max - data_min) * (range_max - range_min) + range_min

    actual_data = normalize(actual_data)
    predicted_data = normalize(predicted_data)
    error_data = normalize(error_data)


    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(18, 6))

    # Actual Velocity Contour Plot
    plt.subplot(131)
    plt.tricontourf(x_coords, y_coords, actual_data, levels=20, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Velocity')
    plt.title('Actual Velocity Contour')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Predicted Velocity Contour Plot
    plt.subplot(132)
    plt.tricontourf(x_coords, y_coords, predicted_data, levels=20, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Velocity')
    plt.title('Predicted Velocity Contour')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Error Contour Plot
    plt.subplot(133)
    plt.tricontourf(x_coords, y_coords, error_data, levels=20, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Normalized Absolute Error')
    plt.title('Velocity Error Contour')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_contours.png'))
    plt.close()

    # Error analysis
    print("Velocity Error Analysis:")
    print(f"Mean Absolute Error: {error_data.mean():.4f}")
    print(f"Median Absolute Error: {np.median(error_data):.4f}")
    print(f"Max Absolute Error: {error_data.max():.4f}")
    print(f"Min Absolute Error: {error_data.min():.4f}")


# def plot_velocity_contours(x_coords, y_coords, actual_data, predicted_data, error_data, output_dir):
#     """
#     Create contour plots for actual velocity, predicted velocity, and error for unstructured data,
#     normalized to a range of -1 to 1 with a single horizontal colorbar.

#     Args:
#         x_coords (np.ndarray): X-coordinates of data points
#         y_coords (np.ndarray): Y-coordinates of data points
#         actual_data (np.ndarray): Actual velocity values
#         predicted_data (np.ndarray): Predicted velocity values
#         error_data (np.ndarray): Absolute error between actual and predicted
#         output_dir (str): Directory to save output plots
#     """
#     # Normalize data to range [-1, 1]
#     def normalize(data, range_min=-1, range_max=1):
#         data_min = data.min()
#         data_max = data.max()
#         return (data - data_min) / (data_max - data_min) * (range_max - range_min) + range_min

#     actual_data = normalize(actual_data)
#     predicted_data = normalize(predicted_data)
#     error_data = normalize(error_data)

#     os.makedirs(output_dir, exist_ok=True)

#     fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

#     # Contour plots
#     cmap = 'coolwarm'
#     levels = 20

#     # Actual Velocity Contour Plot
#     contour1 = axs[0].tricontourf(x_coords, y_coords, actual_data, levels=levels, cmap=cmap)
#     axs[0].set_title('Actual Velocity Contour')
#     axs[0].set_xlabel('X')
#     axs[0].set_ylabel('Y')

#     # Predicted Velocity Contour Plot
#     contour2 = axs[1].tricontourf(x_coords, y_coords, predicted_data, levels=levels, cmap=cmap)
#     axs[1].set_title('Predicted Velocity Contour')
#     axs[1].set_xlabel('X')

#     # Error Contour Plot
#     contour3 = axs[2].tricontourf(x_coords, y_coords, error_data, levels=levels, cmap=cmap)
#     axs[2].set_title('Velocity Error Contour')
#     axs[2].set_xlabel('X')

#     # Horizontal colorbar
#     cbar = fig.colorbar(contour1, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1)
#     cbar.set_label('Normalized Value')

#     plt.savefig(os.path.join(output_dir, 'velocity_contours.png'))
#     plt.close()

#     # Error analysis
#     print("Velocity Error Analysis:")
#     print(f"Mean Absolute Error: {error_data.mean():.4f}")
#     print(f"Median Absolute Error: {np.median(error_data):.4f}")
#     print(f"Max Absolute Error: {error_data.max():.4f}")
#     print(f"Min Absolute Error: {error_data.min():.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Latest checkpoint
    checkpoint_dir = './model_checkpoints'
    checkpoints = [f for f in os.listdir(
        checkpoint_dir) if f.startswith('best_model')]
    latest_checkpoint = sorted(checkpoints)[-1]
    model_path = os.path.join(checkpoint_dir, latest_checkpoint)

    folder_path = "./unstructured_data/CSW"
    dataset = UnstructuredDataset(folder_path)

    # Train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, collate_fn=custom_collate)

    # Load model and get results
    x_coords, y_coords, actual_data, predicted_data, error_data = load_model_and_data(
        model_path, test_loader, device)

    # Visualize results
    output_dir = './visualization_results'
    plot_velocity_contours(x_coords, y_coords, actual_data,
                           predicted_data, error_data, output_dir)

    # Find the latest log file
    log_dir = './logs'
    log_files = [f for f in os.listdir(log_dir) if f.startswith('training_')]
    latest_log = sorted(log_files)[-1]
    log_path = os.path.join(log_dir, latest_log)

if __name__ == "__main__":
    main()
