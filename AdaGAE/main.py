import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import AdaGAE, get_weight_initial
from tqdm import tqdm

from .utils import distance, cal_weights_via_CAN, get_Laplacian_from_weights

warnings.filterwarnings("ignore")


class StochasticAdaGAEModel(nn.Module):
    def __init__(self, config, device):
        super(StochasticAdaGAEModel, self).__init__()
        self.config = config
        self.device = device
        self.lam = config["lam"]
        self.learning_rate = config["learning_rate"]
        self.max_iter = config["max_iter"]
        self.max_epoch = config["max_epoch"]
        self.num_neighbors = config["num_neighbors"] + 1
        self.embedding_dim = config["layers"][-1]
        self.mid_dim = config["layers"][1]
        self.input_dim = config["layers"][0]
        self.update = config["update"]
        self.inc_neighbors = config["inc_neighbors"]
        self.links = config["links"]
        self.embedding = None
        self._build_up()

    def cal_max_neighbors(self, X, labels):
        if not self.update:
            return 0
        size = X.shape[0]
        num_clusters = np.unique(labels).shape[0]
        return 1.0 * size / num_clusters

    def _build_up(self):
        self.W1 = get_weight_initial([self.input_dim, self.mid_dim])
        self.W2 = get_weight_initial([self.mid_dim, self.embedding_dim])

    def update_graph(self):
        weights, raw_weights = cal_weights_via_CAN(self.embedding.t(), self.num_neighbors, self.links)  # first
        weights = weights.detach()
        raw_weights = raw_weights.detach()
        Laplacian = get_Laplacian_from_weights(weights)
        return weights, Laplacian, raw_weights

    def forward(self, X, Laplacian):
        # sparse
        embedding = Laplacian.mm(X.matmul(self.W1))
        embedding = torch.nn.functional.relu(embedding)
        # sparse
        self.embedding = Laplacian.mm(embedding.matmul(self.W2))
        distances = distance(self.embedding.t(), self.embedding.t())
        softmax = torch.nn.Softmax(dim=1)
        recons_w = softmax(-distances)
        return recons_w + 10**-10


class StochasticAdaGAETrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = StochasticAdaGAEModel(config, device).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])

    def build_loss(self, X, embedding, recons, weights, raw_weights):
        size = X.shape[0]
        loss = 0
        loss += raw_weights * torch.log(raw_weights / recons + 10**-10)
        loss = loss.sum(dim=1)
        loss = loss.mean()
        # L2-Regularization
        # loss += 10**-3 * (torch.mean(self.embedding.pow(2)))
        # loss += 10**-3 * (torch.mean(self.W1.pow(2)) + torch.mean(self.W2.pow(2)))
        # loss += 10**-3 * (torch.mean(self.W1.abs()) + torch.mean(self.W2.abs()))
        degree = weights.sum(dim=1)
        L = torch.diag(degree) - weights
        loss += self.config["lam"] * torch.trace(embedding.t().matmul(L).matmul(embedding)) / size
        return loss

    def train(self, X, labels):
        weights, raw_weights = cal_weights_via_CAN(X.t(), self.model.num_neighbors, self.model.links)
        Laplacian = get_Laplacian_from_weights(weights)
        Laplacian = Laplacian.to_sparse()
        torch.cuda.empty_cache()

        dataset = TensorDataset(X, labels)
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)

        pbar = tqdm(range(self.config["epoch"]))
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            epoch_loss = 0.0
            num_batches = 0

            for batch_data in dataloader:
                batch_X, batch_labels = batch_data
                max_neighbors = self.model.cal_max_neighbors(batch_X, batch_labels)
                self.optimizer.zero_grad()
                recons = self.model(Laplacian)
                loss = self.build_loss(batch_X, self.model.embedding, recons, weights, raw_weights)
                loss.backward()
                self.optimizer.step()

                if self.model.num_neighbors < self.model.cal_max_neighbors(batch_X, batch_labels):
                    weights, Laplacian, raw_weights = self.model.update_graph()
                    self.model.num_neighbors += self.model.inc_neighbors
                else:
                    if self.model.update:
                        self.model.num_neighbors = int(max_neighbors)
                        break
                    recons = None
                    weights = weights.cpu()
                    raw_weights = raw_weights.cpu()
                    torch.cuda.empty_cache()
                    w, _, __ = self.model.update_graph()
                    _, __ = (None, None)
                    torch.cuda.empty_cache()
                    weights = weights.to(self.device)
                    raw_weights = raw_weights.to(self.device)
                    if self.model.update:
                        break

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            pbar.set_postfix(loss="{:.3f}".format(avg_loss))
            pbar.update()

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
        },
        "layers": [128, 64],
        "batch_size": 2000,
        "num_neighbors": 5,
        "lam": 0.01,
        "max_iter": 50,
        "max_epoch": 10,
        "update": True,
        "learning_rate": 5e-3,
        "inc_neighbors": 5,
        "lr": 0.01,
        "epoch": 200,
    }
    DATASET_NAME = "fashion-mnist"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_data = torch.load(
        config["dataset"][DATASET_NAME]["feature_path"], map_location="cpu"
    )
    full_labels = torch.load(
        config["dataset"][DATASET_NAME]["label_path"], map_location="cpu"
    ).squeeze()

    n_clusters = len(torch.unique(full_labels))
    config["n_classes"] = n_clusters
    input_dim = full_data.shape[1]

    layers = [input_dim] + config["layers"]

    gae = AdaGAE(
        full_data,
        full_labels,
        layers=layers,
        num_neighbors=config["num_neighbors"],
        lam=config["lam"],
        max_iter=config["max_iter"],
        max_epoch=config["max_epoch"],
        update=config["update"],
        learning_rate=config["learning_rate"],
        inc_neighbors=config["inc_neighbors"],
        device=device,
    )
    )
