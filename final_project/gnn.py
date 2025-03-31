import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
import logging
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


def setup_logging():
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_data(file_path):
    data = np.loadtxt(file_path, skiprows=6)
    x = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]
    return x, y, u


def create_knn_graph(points, k=6):
    """
    Create a k-nearest neighbors graph using scikit-learn.

    Args:
        points (numpy.ndarray): Input points with shape (n_samples, n_features)
        k (int, optional): Number of nearest neighbors. Defaults to 6.

    Returns:
        tuple: (edge_index, num_nodes)
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
    edges = nbrs.kneighbors_graph(points).tocoo()
    edge_index = torch.tensor(
        np.array([edges.row, edges.col]), dtype=torch.long)

    return edge_index, points.shape[0]


def custom_collate(batch):
    """
    Custom collate function to handle graph data with varying sizes.

    Args:
        batch (list): List of (edge_index, node_features) tuples

    Returns:
        Batch: A batched graph data object
    """
    edge_indices, node_features = zip(*batch)

    # PyTorch Geometric Data objects
    data_list = []
    for edge_index, features in zip(edge_indices, node_features):
        data_list.append(Data(x=features, edge_index=edge_index))

    # Single batched graph
    return Batch.from_data_list(data_list)


class VelocityDataset(Dataset):
    def __init__(self, folder_path, k=6, normalize=True):
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith(".dat")]
        self.k = k
        self.data = []
        # self.scaler = MinMaxScaler() if normalize else None
        self.scaler = MinMaxScaler(
            feature_range=(-1, 1)) if normalize else None

        for file in self.files:
            x, y, u = parse_data(os.path.join(folder_path, file))
            points = np.column_stack((x, y))  # Coordinates for KNN

            if normalize and len(u) > 0:
                u = self.scaler.fit_transform(u.reshape(-1, 1)).flatten()

            edge_index, _ = create_knn_graph(points, k=self.k)  # KNN graph
            node_features = torch.from_numpy(u.astype(np.float32)).view(-1, 1)
            self.data.append((edge_index, node_features))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class UnstructuredDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.files = [f for f in os.listdir(data_folder) if f.endswith(".pt")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_folder, self.files[idx])
        edge_index, node_features = torch.load(file_path, weights_only=True)

        return edge_index, node_features


class GraphConvAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_size=32, skip=False):
        super(GraphConvAutoencoder, self).__init__()

        self.latent_size = latent_size
        self.skip = skip

        # Encoder (GCN layers)
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)

        # Bottleneck
        self.fc_encode = nn.Linear(64, latent_size)

        # Decoder layers
        self.fc_decode = nn.Linear(latent_size, 64)

        # Decoder GCN layers (reconstruction)
        self.deconv1 = GCNConv(64 + 32 if skip else 64, 32)
        self.deconv2 = GCNConv(32 + 16 if skip else 32, 16)
        self.deconv3 = GCNConv(16, in_channels)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Custom weight initialization to break symmetry
        """
        for conv in [self.conv1, self.conv2, self.conv3, self.deconv1, self.deconv2, self.deconv3]:
            if hasattr(conv, 'lin') and conv.lin is not None:
                torch.nn.init.xavier_normal_(conv.lin.weight)
                if conv.lin.bias is not None:
                    torch.nn.init.zeros_(conv.lin.bias)

        for fc in [self.fc_encode, self.fc_decode]:
            torch.nn.init.xavier_normal_(fc.weight)
            if fc.bias is not None:
                torch.nn.init.zeros_(fc.bias)

    def forward(self, data):
        # Encoder
        x1 = F.relu(self.conv1(data.x, data.edge_index))
        x1 = self.dropout(x1)

        x2 = F.relu(self.conv2(x1, data.edge_index))
        x2 = self.dropout(x2)

        x3 = F.relu(self.conv3(x2, data.edge_index))
        x3 = self.dropout(x3)

        # Bottleneck
        encode = self.fc_encode(x3)

        # Decoder
        x_dec = self.fc_decode(encode)
        x_dec = F.relu(x_dec)

        # Decoder with optional skip connections
        if self.skip:
            x_dec = torch.cat((x_dec, x2), dim=1)
        x_dec = F.relu(self.deconv1(x_dec, data.edge_index))
        x_dec = self.dropout(x_dec)

        if self.skip:
            x_dec = torch.cat((x_dec, x1), dim=1)
        x_dec = F.relu(self.deconv2(x_dec, data.edge_index))
        x_dec = self.dropout(x_dec)

        # Final reconstruction layer
        x_dec = self.deconv3(x_dec, data.edge_index)

        return torch.tanh(x_dec)  # [-1, 1]
    

class GraphAttentionAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_size=32, skip=False):
        super(GraphAttentionAutoencoder, self).__init__()

        self.latent_size = latent_size
        self.skip = skip

        # Encoder with GAT layers
        self.conv1 = GATConv(in_channels, 16, heads=4, concat=True)  # Multi-head attention
        self.conv2 = GATConv(16 * 4, 32, heads=2, concat=True)       # Reduce heads
        self.conv3 = GATConv(32 * 2, 64, heads=1, concat=True)       # Single head

        # Bottleneck
        self.fc_encode = nn.Linear(64, latent_size)

        # Decoder layers
        self.fc_decode = nn.Linear(latent_size, 64)

        # Decoder GAT layers (reconstruction)
        self.deconv1 = GATConv(64 + 32 if skip else 64, 32, heads=1, concat=True)
        self.deconv2 = GATConv(32 + 16 if skip else 32, 16, heads=1, concat=True)
        self.deconv3 = GATConv(16, in_channels, heads=1, concat=False)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Custom weight initialization to break symmetry
        """
        for module in [self.conv1, self.conv2, self.conv3, self.deconv1, self.deconv2, self.deconv3]:
            module.reset_parameters()

        for fc in [self.fc_encode, self.fc_decode]:
            nn.init.xavier_normal_(fc.weight)
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)

    def forward(self, data):
        # Encoder
        x1 = F.relu(self.conv1(data.x, data.edge_index))
        x1 = self.dropout(x1)

        x2 = F.relu(self.conv2(x1, data.edge_index))
        x2 = self.dropout(x2)

        x3 = F.relu(self.conv3(x2, data.edge_index))
        x3 = self.dropout(x3)

        # Bottleneck
        encode = self.fc_encode(x3)

        # Decoder
        x_dec = self.fc_decode(encode)
        x_dec = F.rrelu(x_dec)

        # Decoder with optional skip connections
        if self.skip:
            x_dec = torch.cat((x_dec, x2), dim=1)
        x_dec = F.relu(self.deconv1(x_dec, data.edge_index))
        x_dec = self.dropout(x_dec)

        if self.skip:
            x_dec = torch.cat((x_dec, x1), dim=1)
        x_dec = F.relu(self.deconv2(x_dec, data.edge_index))
        x_dec = self.dropout(x_dec)

        # Final reconstruction layer
        x_dec = self.deconv3(x_dec, data.edge_index)

        return torch.tanh(x_dec)  # Constrain output to [-1, 1]

def total_variation_loss(x):
    """
    Compute Total Variation Loss for graph-structured data

    Args:
        reconstructed_x (torch.Tensor): Reconstructed node features
        data (torch_geometric.data.Data): Graph data object

    Returns:
        torch.Tensor: Total Variation Loss
    """
    # 4D tensor (batch_size, channels, height, width)
    if x.dim() == 4:
        diff_x = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_y = x[:, :, :, 1:] - x[:, :, :, :-1]
        tv_loss = torch.sum(torch.abs(diff_x)) + torch.sum(torch.abs(diff_y))
    # 2D tensor (batch_size, features)
    elif x.dim() == 2:
        diff_x = x[:, 1:] - x[:, :-1]
        tv_loss = torch.sum(torch.abs(diff_x))
    else:
        raise ValueError(f"Unsupported tensor shape: {x.shape}")

    return tv_loss


def train_and_validate(model, train_loader, val_loader, device, logger, checkpoint_dir, num_epochs=50, patience=10, tv_weight=0.0):

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    # Training
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

    
            reconstructed_x = model(batch)
            mse_loss = F.mse_loss(reconstructed_x, batch.x)
            tv_loss = total_variation_loss(reconstructed_x)

            # Total loss (MSE + TV loss with weight)
            loss = mse_loss + tv_weight * tv_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed_x = model(batch)
                mse_loss = F.mse_loss(reconstructed_x, batch.x)
                tv_loss = total_variation_loss(reconstructed_x)
                loss = mse_loss + tv_weight * tv_loss
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping and model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save the best model
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            logger.info(f"Saved new best model at {checkpoint_path}")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break

    # Losses to CSV
    loss_data = pd.DataFrame({
        'epoch': range(num_epochs),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_data.to_csv(os.path.join(checkpoint_dir, 'losses.csv'), index=False)
    logger.info(f"Losses saved to 'losses.csv' in {checkpoint_dir}")

    # Loss vs epoch plot
    plt.figure()
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss over Epochs')
    plt.savefig(os.path.join(checkpoint_dir, 'losses.png'))
    logger.info(f"Loss plot saved as 'losses.png' in {checkpoint_dir}")

    return model



def main():
    logger = setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    folder_path = './unstructured_data/CSW'
    dataset = UnstructuredDataset(folder_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=32,
                            shuffle=False, collate_fn=custom_collate)

    # Model initialization
    num_features = 1
    model = GraphConvAutoencoder(in_channels=num_features)
    # model = GraphAttentionAutoencoder(in_channels=num_features)

    # Training and validation
    checkpoint_dir = './model_checkpoints'
    trained_model = train_and_validate(
        model,
        train_loader,
        val_loader,
        device,
        logger,
        checkpoint_dir
    )

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    ########################Create Graph Data for training#############################
    #  
    # folder_path = './data/datasets/CSW'
    # dataset = VelocityDataset(folder_path)
    # # dataset = Subset(dataset, range(500))

    # # Save the unstructured data
    # output_folder = './unstructured_data/CSW'
    # os.makedirs(output_folder, exist_ok=True)

    # for i, (edge_index, node_features) in enumerate(dataset):
    #     torch.save((edge_index, node_features), os.path.join(output_folder, f'snapshot_{i}.pt'))

    # print(f"Unstructured data saved to {output_folder}")

    #####################################################################################

    ######################################Train model####################################
    main()
