import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from dekm import DEKMDenseModel


class DEKMDenseTrainer:
    def __init__(
        self,
        input_shape,
        hidden_units,
        n_clusters,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the DEKM Dense Trainer

        Args:
            input_shape (int): Dimension of input features
            hidden_units (int): Dimension of hidden/encoded features
            n_clusters (int): Number of clusters
            device (str): Device to run on ('cuda' or 'cpu')
        """
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.n_clusters = n_clusters
        self.device = device
        # Initialize model
        self.model = DEKMDenseModel(input_shape, hidden_units).to(device)

        # Training parameters
        self.pretrain_epochs = 200
        self.pretrain_batch_size = 256
        self.batch_size = 256
        self.update_interval = 10

    def pretrain_loss(self, y_pred, y_true):
        """
        Pretraining loss function - reconstruction loss
        Equivalent to TensorFlow's loss_train_base function
        """
        # Extract reconstruction part (skip hidden units part)
        y_pred_recon = y_pred[:, self.hidden_units :]
        return nn.MSELoss()(y_pred_recon, y_true)

    def pretrain(self, x, save_path=None):
        """
        Pretrain the autoencoder
        Equivalent to TensorFlow's train_base function

        Args:
            x (torch.Tensor): Input data
            save_path (str): Path to save pretrained weights
        """
        print("Starting pretraining...")

        # Create dataset and dataloader
        dataset = TensorDataset(x, x)  # Input and target are the same for autoencoder
        dataloader = DataLoader(
            dataset, batch_size=self.pretrain_batch_size, shuffle=True
        )

        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters())

        # Training loop
        self.model.train()
        for epoch in range(self.pretrain_epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = self.pretrain_loss(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 50 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")

        if save_path:
            torch.save(self.model.state_dict(), save_path)
            print(f"Pretrained weights saved to {save_path}")

        print("Pretraining completed!")

    def sorted_eig(self, X):
        """
        Compute sorted eigenvalues and eigenvectors
        Equivalent to TensorFlow's sorted_eig function
        """
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        e_vals, e_vecs = np.linalg.eig(X_np)
        idx = np.argsort(e_vals)
        e_vecs = e_vecs[:, idx]
        e_vals = e_vals[idx]
        return e_vals, e_vecs

    def train(self, x, y):
        """
        Main DEKM training function
        Equivalent to TensorFlow's train function

        Args:
            x (torch.Tensor): Input data
            y (torch.Tensor): True labels (for evaluation)
        """
        print("Starting DEKM training...")

        # Setup
        optimizer = optim.Adam(self.model.parameters())
        loss_value = 0
        index = 0
        kmeans_n_init = 100
        assignment = np.array([-1] * len(x))
        index_array = np.arange(x.shape[0])

        # Move data to device
        x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)

        self.model.train()

        # Main training loop
        for ite in range(int(140 * 100)):  # 14000 iterations
            # Update cluster assignments and compute eigenvectors
            if ite % self.update_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    H = self.model(x)[:, : self.hidden_units].cpu().numpy()

                # K-means clustering
                ans_kmeans = KMeans(
                    n_clusters=self.n_clusters, n_init=kmeans_n_init, random_state=42
                ).fit(H)
                kmeans_n_init = max(10, int(ans_kmeans.n_iter_ * 2))  # Adaptive n_init

                U = ans_kmeans.cluster_centers_
                assignment_new = ans_kmeans.labels_

                # Hungarian algorithm for assignment matching
                w = np.zeros((self.n_clusters, self.n_clusters), dtype=np.int64)
                for i in range(len(assignment_new)):
                    if assignment[i] != -1:  # Skip initial assignments
                        w[assignment_new[i], assignment[i]] += 1

                if ite > 0:  # Skip first iteration
                    row_ind, col_ind = linear_sum_assignment(-w)
                    temp = np.array(assignment)
                    for i in range(self.n_clusters):
                        assignment[temp == col_ind[i]] = i

                assignment = assignment_new

                # Compute within-cluster scatter matrices
                S_i = []
                for i in range(self.n_clusters):
                    cluster_points = H[assignment == i]
                    if len(cluster_points) > 0:
                        temp = cluster_points - U[i]
                        scatter = np.matmul(temp.T, temp)
                        S_i.append(scatter)
                    else:
                        S_i.append(np.zeros((self.hidden_units, self.hidden_units)))

                S_i = np.array(S_i)
                S = np.sum(S_i, 0)

                # Eigenvalue decomposition
                Evals, V = self.sorted_eig(S)
                H_vt = np.matmul(H, V)
                U_vt = np.matmul(U, V)

                # Convert back to tensors
                V_tensor = torch.FloatTensor(V).to(self.device)
                H_vt_tensor = torch.FloatTensor(H_vt).to(self.device)
                U_vt_tensor = torch.FloatTensor(U_vt).to(self.device)

                self.model.train()

            # Mini-batch training
            idx = index_array[
                index * self.batch_size : min((index + 1) * self.batch_size, x.shape[0])
            ]

            if len(idx) == 0:
                index = 0
                continue

            # Prepare target
            y_true = H_vt_tensor[idx].clone()
            temp = assignment[idx]
            for i in range(len(idx)):
                if temp[i] >= 0:  # Valid assignment
                    y_true[i, -1] = U_vt_tensor[temp[i], -1]

            # Forward pass and loss computation
            optimizer.zero_grad()
            y_pred = self.model(x[idx])
            y_pred_cluster = torch.matmul(y_pred[:, : self.hidden_units], V_tensor)

            loss_value = nn.MSELoss()(y_pred_cluster, y_true)
            loss_value.backward()
            optimizer.step()

            # Update batch index
            index = index + 1 if (index + 1) * self.batch_size <= x.shape[0] else 0

            # Early stopping condition (optional)
            if ite > 0 and ite % 1000 == 0:
                print(f"Iteration {ite}, Loss: {loss_value.item():.6f}")

        print("DEKM training completed!")
        return assignment

    def save_model(self, path):
        """Save the trained model"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load a pre-trained model"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")

    def predict_clusters(self, x):
        """
        Predict cluster assignments for given data

        Args:
            x (torch.Tensor): Input data

        Returns:
            np.ndarray: Cluster assignments
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            H = self.model(x)[:, : self.hidden_units].cpu().numpy()

        # Use final KMeans to get clusters
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        assignments = kmeans.fit_predict(H)

        return assignments
