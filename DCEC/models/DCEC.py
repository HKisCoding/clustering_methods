import csv
import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .ConvAE import ConvAutoEncoder


class ClusteringLayer(nn.Module):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents
    the probability of the sample belonging to each cluster. The probability is calculated with
    student's t-distribution.

    Args:
        n_clusters: number of clusters.
        alpha: parameter in Student's t-distribution. Default to 1.0.
        initial_weights: initial cluster centers with shape (n_clusters, n_features)

    Input shape:
        2D tensor with shape: (n_samples, n_features).
    Output shape:
        2D tensor with shape: (n_samples, n_clusters).
    """

    def __init__(self, n_clusters, alpha=1.0, initial_weights=None):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = initial_weights

        # This will be initialized later when we know the input dimension
        self.clusters = None

    def build(self, input_dim):
        """Initialize the cluster centers"""
        self.clusters = nn.Parameter(torch.Tensor(self.n_clusters, input_dim))
        nn.init.xavier_uniform_(self.clusters)

        if self.initial_weights is not None:
            self.clusters.data = torch.tensor(self.initial_weights, dtype=torch.float32)

    def forward(self, inputs):
        """
        Student t-distribution, as same as used in t-SNE algorithm.
        q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.

        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        if self.clusters is None:
            raise RuntimeError("ClusteringLayer not built. Call build() first.")

        # Calculate squared euclidean distance between inputs and cluster centers
        # inputs: (n_samples, n_features)
        # clusters: (n_clusters, n_features)
        # Expand dims: inputs -> (n_samples, 1, n_features), clusters -> (1, n_clusters, n_features)
        inputs_expanded = inputs.unsqueeze(1)  # (n_samples, 1, n_features)
        clusters_expanded = self.clusters.unsqueeze(0)  # (1, n_clusters, n_features)

        # Calculate squared distances
        distances = torch.sum(
            (inputs_expanded - clusters_expanded) ** 2, dim=2
        )  # (n_samples, n_clusters)

        # Student t-distribution
        q = 1.0 / (1.0 + distances / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)

        # Normalize to get probabilities
        q = q / torch.sum(q, dim=1, keepdim=True)

        return q

    def set_weights(self, weights):
        """Set cluster centers"""
        if self.clusters is None:
            raise RuntimeError("ClusteringLayer not built. Call build() first.")
        self.clusters.data = torch.tensor(weights[0], dtype=torch.float32)


class DCEC(nn.Module):
    """
    Deep Convolutional Embedded Clustering (DCEC) model
    """

    def __init__(
        self, input_shape, filters=[32, 64, 128, 10], n_clusters=10, alpha=1.0
    ):
        super(DCEC, self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []

        # Create the convolutional autoencoder
        self.cae = ConvAutoEncoder(input_shape, filters)

        # Create clustering layer
        self.clustering_layer = ClusteringLayer(n_clusters, alpha)

        # Initialize clustering layer after we know the embedding dimension
        self.clustering_layer.build(filters[3])  # embedding dimension

        # For extracting features (encoder part)
        self.encoder = None

    def extract_features(self, x):
        """Extract features from the embedding layer"""
        return self.cae.encode(x)

    def forward(self, x):
        """
        Forward pass of DCEC
        Returns:
            q: cluster assignments (soft labels)
            reconstructed: reconstructed input
        """
        # Get features from encoder
        features = self.cae.encode(x)

        # Get cluster assignments
        q = self.clustering_layer(features)

        # Get reconstructed input
        reconstructed = self.cae.decode(features)

        return q, reconstructed

    def predict(self, x):
        """Predict cluster labels"""
        self.eval()
        with torch.no_grad():
            q, _ = self.forward(x)
            return q.argmax(1).cpu().numpy()

    @staticmethod
    def target_distribution(q):
        """
        Calculate target distribution P
        """
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def pretrain(
        self,
        x,
        batch_size=256,
        epochs=200,
        lr=0.001,
        save_dir="results/temp",
        device="cpu",
    ):
        """
        Pretrain the convolutional autoencoder
        """
        print("...Pretraining...")

        # Move model and data to device
        self.cae = self.cae.to(device)
        x = x.to(device)

        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.cae.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Create data loader
        dataset = torch.utils.data.TensorDataset(x, x)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Training loop
        self.cae.train()
        t0 = time()

        # Create CSV logger
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        csv_file = open(os.path.join(save_dir, "pretrain_log.csv"), "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "loss"])

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch_x, batch_target in dataloader:
                optimizer.zero_grad()

                # Forward pass
                output = self.cae(batch_x)
                loss = criterion(output, batch_target)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            csv_writer.writerow([epoch, avg_loss])

            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.6f}")

        csv_file.close()
        print("Pretraining time: ", time() - t0)

        # Save pretrained model
        model_path = os.path.join(save_dir, "pretrain_cae_model.pth")
        torch.save(self.cae.state_dict(), model_path)
        print(f"Pretrained weights are saved to {model_path}")

        self.pretrained = True

    def load_weights(self, weights_path, device="cpu"):
        """Load model weights"""
        state_dict = torch.load(weights_path, map_location=device)
        self.load_state_dict(state_dict)

    def fit(
        self,
        x,
        y=None,
        batch_size=256,
        maxiter=20000,
        tol=1e-3,
        update_interval=140,
        cae_weights=None,
        save_dir="./results/temp",
        lr=0.001,
        device="cpu",
    ):
        """
        Train the DCEC model
        """
        print("Update interval", update_interval)
        save_interval = x.shape[0] // batch_size * 5
        print("Save interval", save_interval)

        # Move to device
        self = self.to(device)
        x = x.to(device)
        if y is not None:
            y = (
                y.to(device)
                if isinstance(y, torch.Tensor)
                else torch.tensor(y).to(device)
            )

        # Step 1: pretrain if necessary
        t0 = time()
        if not self.pretrained and cae_weights is None:
            print("...pretraining CAE using default hyper-parameters:")
            print("   optimizer=Adam; epochs=200")
            self.pretrain(
                x, batch_size, epochs=200, lr=lr, save_dir=save_dir, device=device
            )
            self.pretrained = True
        elif cae_weights is not None:
            self.cae.load_state_dict(torch.load(cae_weights, map_location=device))
            print("cae_weights is loaded successfully.")

        # Step 2: initialize cluster centers using k-means
        t1 = time()
        print("Initializing cluster centers with k-means.")

        self.eval()
        with torch.no_grad():
            features = self.extract_features(x).cpu().numpy()

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        self.y_pred = kmeans.fit_predict(features)
        y_pred_last = np.copy(self.y_pred)

        # Set initial cluster centers
        self.clustering_layer.set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # Setup logging
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        logfile = open(os.path.join(save_dir, "dcec_log.csv"), "w", newline="")
        fieldnames = ["iter", "acc", "nmi", "ari", "L", "Lc", "Lr"]
        logwriter = csv.DictWriter(logfile, fieldnames=fieldnames)
        logwriter.writeheader()

        # Setup optimizer for full model
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Loss functions
        kld_loss = nn.KLDivLoss(size_average=False)
        mse_loss = nn.MSELoss()

        t2 = time()
        loss = [0, 0, 0]
        index = 0

        # Create data indices for batch sampling
        n_samples = x.shape[0]
        indices = np.arange(n_samples)

        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                # Update target distribution
                self.eval()
                with torch.no_grad():
                    q, _ = self.forward(x)
                    q = q.cpu().numpy()

                p = self.target_distribution(torch.tensor(q)).numpy()
                p = torch.tensor(p, dtype=torch.float32).to(device)

                # Evaluate clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    try:
                        # Import metrics (assuming they exist)
                        import metrics

                        y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
                        acc = np.round(metrics.acc(y_np, self.y_pred), 5)
                        nmi = np.round(metrics.nmi(y_np, self.y_pred), 5)
                        ari = np.round(metrics.ari(y_np, self.y_pred), 5)
                        loss_rounded = np.round(loss, 5)
                        logdict = dict(
                            iter=ite,
                            acc=acc,
                            nmi=nmi,
                            ari=ari,
                            L=loss_rounded[0],
                            Lc=loss_rounded[1],
                            Lr=loss_rounded[2],
                        )
                        logwriter.writerow(logdict)
                        print(
                            f"Iter {ite}: Acc {acc}, nmi {nmi}, ari {ari}; loss={loss_rounded}"
                        )
                    except ImportError:
                        print(f"Iter {ite}: loss={np.round(loss, 5)}")

                # Check stop criterion
                delta_label = (
                    np.sum(self.y_pred != y_pred_last).astype(np.float32)
                    / self.y_pred.shape[0]
                )
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print(f"delta_label {delta_label} < tol {tol}")
                    print("Reached tolerance threshold. Stopping training.")
                    logfile.close()
                    break

            # Train on batch
            self.train()

            # Sample batch indices
            if (index + 1) * batch_size > n_samples:
                batch_indices = indices[index * batch_size :]
                index = 0
            else:
                batch_indices = indices[index * batch_size : (index + 1) * batch_size]
                index += 1

            batch_x = x[batch_indices]
            batch_p = p[batch_indices]

            optimizer.zero_grad()

            # Forward pass
            q_batch, x_reconstructed = self.forward(batch_x)

            # Calculate losses
            # Clustering loss (KL divergence)
            kld = kld_loss(torch.log(q_batch), batch_p) / batch_x.shape[0]

            # Reconstruction loss
            mse = mse_loss(x_reconstructed, batch_x)

            # Total loss
            total_loss = kld + mse

            # Backward pass
            total_loss.backward()
            optimizer.step()

            loss = [total_loss.item(), kld.item(), mse.item()]

            # Save intermediate model
            if ite % save_interval == 0:
                print(f"saving model to: {save_dir}/dcec_model_{ite}.pth")
                torch.save(self.state_dict(), f"{save_dir}/dcec_model_{ite}.pth")

            ite += 1

        # Save the trained model
        logfile.close()
        print(f"saving model to: {save_dir}/dcec_model_final.pth")
        torch.save(self.state_dict(), f"{save_dir}/dcec_model_final.pth")

        t3 = time()
        print("Pretrain time:  ", t1 - t0)
        print("Clustering time:", t3 - t1)
        print("Total time:     ", t3 - t0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DCEC")
    parser.add_argument("--dataset", default="usps", choices=["mnist", "usps"])
    parser.add_argument("--n_clusters", default=10, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--maxiter", default=20000, type=int)
    parser.add_argument("--tol", default=0.001, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--update_interval", default=140, type=int)
    parser.add_argument("--pretrain_epochs", default=200, type=int)
    parser.add_argument("--save_dir", default="results/temp", type=str)
    parser.add_argument(
        "--cae_weights", default=None, type=str, help="Path to pretrained CAE weights"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str
    )
    args = parser.parse_args()

    print(args)

    # Load dataset
    import sys

    sys.path.append("..")
    from datasets import load_mnist, load_usps

    if args.dataset == "mnist":
        x, y = load_mnist()
    elif args.dataset == "usps":
        x, y = load_usps("data/usps")

    # Convert to PyTorch tensors and adjust shape
    if len(x.shape) == 4:
        x = torch.FloatTensor(x).permute(0, 3, 1, 2)  # Convert to (N, C, H, W)
    else:
        x = torch.FloatTensor(x).unsqueeze(1)  # Add channel dimension

    if isinstance(y, np.ndarray):
        y = torch.LongTensor(y)

    # Create model
    input_shape = x.shape[1:]  # (C, H, W)
    model = DCEC(
        input_shape=input_shape, filters=[32, 64, 128, 10], n_clusters=args.n_clusters
    )

    print(f"Model created with input shape: {input_shape}")

    # Train the model
    model.fit(
        x=x,
        y=y,
        batch_size=args.batch_size,
        maxiter=args.maxiter,
        tol=args.tol,
        update_interval=args.update_interval,
        cae_weights=args.cae_weights,
        save_dir=args.save_dir,
        lr=args.lr,
        device=args.device,
    )
