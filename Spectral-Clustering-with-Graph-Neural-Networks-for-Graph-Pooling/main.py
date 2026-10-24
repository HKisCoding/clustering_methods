import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.neighbors import kneighbors_graph
from torch.nn import Linear
from torch_geometric import utils
from torch_geometric.nn import GraphConv, Sequential, dense_mincut_pool
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import common


class Net(torch.nn.Module):
    def __init__(
        self,
        mp_units,
        mp_act,
        in_channels,
        n_clusters,
        mlp_units=[],
        mlp_act="Identity",
    ):
        super().__init__()

        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)

        # Message passing layers
        mp = [
            (GraphConv(in_channels, mp_units[0]), "x, edge_index, edge_weight -> x"),
            mp_act,
        ]
        for i in range(len(mp_units) - 1):
            mp.append(
                (
                    GraphConv(mp_units[i], mp_units[i + 1]),
                    "x, edge_index, edge_weight -> x",
                )
            )
            mp.append(mp_act)
        self.mp = Sequential("x, edge_index, edge_weight", mp)
        out_chan = mp_units[-1]

        # MLP layers
        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))

    def forward(self, x, edge_index, edge_weight):
        # Propagate node feats
        x = self.mp(x, edge_index, edge_weight)

        # Cluster assignments (logits)
        s = self.mlp(x)

        # Obtain MinCutPool losses
        adj = utils.to_dense_adj(
            edge_index, edge_attr=edge_weight, max_num_nodes=x.size(0)
        )
        _, _, mc_loss, o_loss = dense_mincut_pool(x, adj, s)

        return torch.softmax(s, dim=-1), mc_loss, o_loss


class Trainer:
    def __init__(self, config, device):
        self.device = device
        self.lr = config.get("lr", 1e-2)
        self.epochs = config.get("epochs", 10000)
        self.patience = config.get("patience", 50)
        self.n_clusters = config.get("n_clusters", None)
        self.batch_size = config.get("batch_size", 2000)

        # Initialize model
        self.model = Net(
            mp_units=config.get("mp_units", [16]),
            mp_act=config.get("mp_act", "ELU"),
            in_channels=config.get("in_channels", None),
            n_clusters=config.get("n_clusters", None),
            mlp_units=config.get("mlp_units", []),
            mlp_act=config.get("mlp_act", "Identity"),
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, x, edge_index, edge_weight, labels):
        self.x = x
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.labels = labels

        num_nodes = x.shape[0]
        # Cap the batch size at the dataset size.
        batch_size = min(self.batch_size, num_nodes)

        patience = self.patience
        best_nmi = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            # Shuffle nodes and split into mini-batches each epoch.
            perm = torch.randperm(num_nodes, device=self.device)
            total_loss = 0.0
            num_batches = 0
            for start in range(0, num_nodes, batch_size):
                idx = perm[start : start + batch_size]

                # Induced subgraph over the batch nodes (edges relabeled to
                # the local 0..len(idx)-1 indexing used by the batch features).
                sub_edge_index, sub_edge_weight = utils.subgraph(
                    idx,
                    edge_index,
                    edge_weight,
                    relabel_nodes=True,
                    num_nodes=num_nodes,
                )

                self.optimizer.zero_grad()
                _, mc_loss, o_loss = self.model(x[idx], sub_edge_index, sub_edge_weight)
                loss = mc_loss + o_loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            # Evaluate for early stopping
            if epoch % 10 == 0 or epoch == 1:
                pred = self._predict(batch_size)
                nmi = NMI(pred, self.labels.cpu().numpy())
                print(f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, NMI: {nmi:.3f}")

                if nmi > best_nmi:
                    best_nmi = nmi
                    patience = self.patience
                else:
                    patience -= 1

                if patience == 0:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def _predict(self, batch_size=None):
        """Run batched inference and return cluster assignments for all nodes."""
        num_nodes = self.x.shape[0]
        if batch_size is None:
            batch_size = min(self.batch_size, num_nodes)

        self.model.eval()
        preds = torch.empty(num_nodes, dtype=torch.long)
        with torch.no_grad():
            for start in range(0, num_nodes, batch_size):
                idx = torch.arange(
                    start, min(start + batch_size, num_nodes), device=self.device
                )
                sub_edge_index, sub_edge_weight = utils.subgraph(
                    idx,
                    self.edge_index,
                    self.edge_weight,
                    relabel_nodes=True,
                    num_nodes=num_nodes,
                )
                clust, _, _ = self.model(self.x[idx], sub_edge_index, sub_edge_weight)
                preds[start : start + idx.shape[0]] = clust.max(1)[1].cpu()

        return preds.numpy()

    def evaluate(self, num_clusters):
        pred = self._predict()
        results = common.run_evaluate(pred, self.labels.cpu().numpy(), num_clusters)
        return results


def features_to_graph(features, k_neighbors=10):
    """
    Convert features to graph structure using k-nearest neighbors.

    Args:
        features: torch.Tensor of shape [N, D]
        k_neighbors: number of neighbors for k-NN graph

    Returns:
        edge_index: torch.Tensor of shape [2, E]
        edge_weight: torch.Tensor of shape [E]
    """
    # Convert to numpy for sklearn
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().numpy()
    else:
        features_np = features

    # Create k-NN graph (symmetric mode creates bidirectional edges)
    knn_graph = kneighbors_graph(
        features_np,
        n_neighbors=k_neighbors,
        mode="connectivity",
        include_self=False,
        n_jobs=10,
    )

    # Make symmetric (if i->j exists, ensure j->i exists)
    knn_graph = knn_graph + knn_graph.transpose()
    knn_graph.data = np.ones_like(knn_graph.data)  # Set all edges to 1

    # Convert to edge_index format
    knn_graph = knn_graph.tocoo()
    row = torch.from_numpy(knn_graph.row).long()
    col = torch.from_numpy(knn_graph.col).long()
    edge_index = torch.stack([row, col], dim=0)

    # Edge weights (all 1.0 for connectivity graph)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

    return edge_index, edge_weight


def main():
    config = {
        "dataset": {
            "coil-20": {
                "feature_path": "dataset/embedding/resnet/coil-20_Feature.pt",
                "label_path": "dataset/embedding/resnet/coil-20_Label.pt",
            },
            "MSRC-v2": {
                "feature_path": "dataset/embedding/resnet/MSRC-v2_Feature.pt",
                "label_path": "dataset/embedding/resnet/MSRC-v2_Label.pt",
            },
            "USPS": {
                "feature_path": "dataset/embedding/auto_encoder/USPS_Feature.pt",
                "label_path": "dataset/embedding/auto_encoder/USPS_Label.pt",
            },
            "mnist": {
                "feature_path": "dataset/embedding/auto_encoder/mnist_raw_Feature.pt",
                "label_path": "dataset/embedding/auto_encoder/mnist_raw_Label.pt",
            },
            "fashion-mnist": {
                "feature_path": "dataset/embedding/auto_encoder/fashion_mnist_Feature.pt",
                "label_path": "dataset/embedding/auto_encoder/fashion_mnist_Label.pt",
            },
            "Caltech_101": {
                "feature_path": "dataset/embedding/resnet/Caltech_101_Feature.pt",
                "label_path": "dataset/embedding/resnet/Caltech_101_Label.pt",
            },
        },
        "mp_units": [16],
        "mp_act": "ELU",
        "n_clusters": None,  # Will be set from dataset
        "mlp_units": [],
        "mlp_act": "Identity",
        "lr": 1e-2,
        "epochs": 500,
        "patience": 50,
        "batch_size": 2000,  # Mini-batch size; capped at dataset size
        "k_neighbors": 10,  # Number of neighbors for k-NN graph
    }

    DATASET_NAME = "mnist"

    torch.manual_seed(0)  # for reproducibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load features and labels
    features = torch.load(config["dataset"][DATASET_NAME]["feature_path"])
    labels = torch.load(config["dataset"][DATASET_NAME]["label_path"]).squeeze()

    # Convert to graph structure
    edge_index, edge_weight = features_to_graph(
        features, k_neighbors=config["k_neighbors"]
    )

    # Normalize adjacency matrix
    edge_index, edge_weight = gcn_norm(
        edge_index,
        edge_weight,
        features.shape[0],
        add_self_loops=False,
        dtype=features.dtype,
    )

    # Move to device
    features = features.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # Set config values from dataset
    config["in_channels"] = features.shape[1]
    n_clusters = len(torch.unique(labels))
    config["n_clusters"] = n_clusters

    eval_results = []
    for _ in range(1):
        start_train = time.perf_counter()
        trainer = Trainer(config, device)
        trainer.train(features, edge_index, edge_weight, labels)
        print(f"train: {time.perf_counter() - start_train}")

        start_infer = time.perf_counter()
        results = trainer.evaluate(n_clusters)
        print(f"infer: {time.perf_counter() - start_infer}")
        eval_results.append(results)

        pd.DataFrame(eval_results).to_csv(
            f"output/spectral-GNN-graph-pooling/{DATASET_NAME}_test.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
