import time

import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
import torch.nn.init as init
import utils
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from tqdm import tqdm

import common


class MLP(nn.Module):
    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=False):
        super(MLP, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.output_dims = out_dims
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_dims, self.hid_dims[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hid_dims) - 1):
            self.layers.append(nn.Linear(self.hid_dims[i], self.hid_dims[i + 1]))
            self.layers.append(nn.ReLU())

        self.out_layer = nn.Linear(self.hid_dims[-1], self.output_dims)
        if kaiming_init:
            self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)
        init.xavier_uniform_(self.out_layer.weight)
        init.zeros_(self.out_layer.bias)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        h = self.out_layer(h)
        h = torch.tanh_(h)
        return h


class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter(
            "bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float())
        )

    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)


class SENet(nn.Module):
    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=True):
        super(SENet, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.kaiming_init = kaiming_init
        self.shrink = 1.0 / out_dims

        self.net_q = MLP(
            input_dims=self.input_dims,
            hid_dims=self.hid_dims,
            out_dims=self.out_dims,
            kaiming_init=self.kaiming_init,
        )

        self.net_k = MLP(
            input_dims=self.input_dims,
            hid_dims=self.hid_dims,
            out_dims=self.out_dims,
            kaiming_init=self.kaiming_init,
        )

        self.thres = AdaptiveSoftThreshold(1)

    def query_embedding(self, queries):
        q_emb = self.net_q(queries)
        return q_emb

    def key_embedding(self, keys):
        k_emb = self.net_k(keys)
        return k_emb

    def get_coeff(self, q_emb, k_emb):
        c = self.thres(q_emb.mm(k_emb.t()))
        return self.shrink * c

    def forward(self, queries, keys):
        q = self.query_embedding(queries)
        k = self.key_embedding(keys)
        out = self.get_coeff(q_emb=q, k_emb=k)
        return out


class Trainer:
    def __init__(self, config, device):
        self.gamma = config["gamma"]
        self.lmbd = config["lmbd"]
        self.hid_dims = config["hid_dims"]
        self.out_dims = config["out_dims"]
        self.total_iters = config["total_iters"]
        self.eval_iters = config["eval_iters"]
        self.lr = config["lr"]
        self.lr_min = config["lr_min"]
        self.batch_size = config["batch_size"]
        self.chunk_size = config["chunk_size"]
        self.non_zeros = config["non_zeros"]
        self.n_neighbors = config["n_neighbors"]
        self.spectral_dim = config["spectral_dim"]
        self.affinity = config["affinity"]
        self.mean_subtraction = config["mean_subtraction"]
        self.device = device

    def train(self, features, labels):
        self.features = features
        self.labels = labels
        for N in [200, 500, 1000, 2000, 5000, 10000, 20000]:
            block_size = min(N, self.chunk_size)
            n_iter_per_epoch = features.shape[0] // self.batch_size
            n_step_per_iter = round(features.shape[0] // block_size)
            n_epochs = self.total_iters // n_iter_per_epoch

            self.senet = SENet(
                features.shape[1], self.hid_dims, self.out_dims, kaiming_init=True
            ).to(self.device)
            optimizer = torch.optim.Adam(self.senet.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=self.lr_min
            )

            n_iters = 0
            pbar = tqdm(range(n_epochs))
            for epoch in pbar:
                pbar.set_description(f"Epoch {epoch}")
                randidx = torch.randperm(features.shape[0])

                for i in range(n_iter_per_epoch):
                    self.senet.train()

                    batch_idx = randidx[i * self.batch_size : (i + 1) * self.batch_size]
                    batch = features[batch_idx].to(self.device)
                    q_batch = self.senet.query_embedding(batch)
                    k_batch = self.senet.key_embedding(batch)

                    rec_batch = torch.zeros_like(batch).to(self.device)
                    reg = torch.zeros([1]).to(self.device)
                    for j in range(n_step_per_iter):
                        block = features[j * block_size : (j + 1) * block_size].to(
                            self.device
                        )
                        k_block = self.senet.key_embedding(block)
                        c = self.senet.get_coeff(q_batch, k_block)
                        rec_batch = rec_batch + c.mm(block)
                        reg = reg + regularizer(c, self.lmbd)
                    diag_c = (
                        self.senet.thres((q_batch * k_batch).sum(dim=1, keepdim=True))
                        * self.senet.shrink
                    )
                    rec_batch = rec_batch - diag_c * batch
                    reg = reg - regularizer(diag_c, self.lmbd)
                    rec_loss = torch.sum(torch.pow(batch - rec_batch, 2))
                    loss = (0.5 * self.gamma * rec_loss + reg) / self.batch_size
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.senet.parameters(), 0.001)
                    optimizer.step()

                    n_iters += 1

                pbar.set_postfix(
                    loss="{:3.4f}".format(loss.item()),
                    rec_loss="{:3.4f}".format(rec_loss.item() / self.batch_size),
                    reg="{:3.4f}".format(reg.item() / self.batch_size),
                )
                scheduler.step()

    def evaluate(
        self,
        num_subspaces,
        affinity="nearest_neighbor",
        knn_mode="symmetric",
    ):
        C_sparse = get_sparse_rep(
            senet=self.senet,
            data=self.features,
            batch_size=self.batch_size,
            chunk_size=self.chunk_size,
            non_zeros=self.non_zeros,
        )
        C_sparse_normalized = normalize(C_sparse, return_norm=False)
        if affinity == "symmetric":
            Aff = 0.5 * (
                np.absolute(C_sparse_normalized) + np.absolute(C_sparse_normalized).T
            )
        elif affinity == "nearest_neighbor":
            Aff = get_knn_Aff(C_sparse_normalized, k=self.n_neighbors, mode=knn_mode)
        else:
            raise Exception("affinity should be 'symmetric' or 'nearest_neighbor'")
        preds = utils.spectral_clustering(Aff, num_subspaces, self.spectral_dim)
        results = common.run_evaluate(preds, self.labels.cpu().numpy(), num_subspaces)
        return results


def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()


def get_sparse_rep(senet, data, batch_size=10, chunk_size=100, non_zeros=1000):
    N, D = data.shape
    non_zeros = min(N, non_zeros)
    C = torch.empty([batch_size, N])
    if N % batch_size != 0:
        raise Exception("batch_size should be a factor of dataset size.")
    if N % chunk_size != 0:
        raise Exception("chunk_size should be a factor of dataset size.")

    val = []
    indicies = []
    with torch.no_grad():
        senet.eval()
        for i in range(data.shape[0] // batch_size):
            chunk = data[i * batch_size : (i + 1) * batch_size].cuda()
            q = senet.query_embedding(chunk)
            for j in range(data.shape[0] // chunk_size):
                chunk_samples = data[j * chunk_size : (j + 1) * chunk_size].cuda()
                k = senet.key_embedding(chunk_samples)
                temp = senet.get_coeff(q, k)
                C[:, j * chunk_size : (j + 1) * chunk_size] = temp.cpu()

            rows = list(range(batch_size))
            cols = [j + i * batch_size for j in rows]
            C[rows, cols] = 0.0

            _, index = torch.topk(torch.abs(C), dim=1, k=non_zeros)

            val.append(C.gather(1, index).reshape([-1]).cpu().data.numpy())
            index = index.reshape([-1]).cpu().data.numpy()
            indicies.append(index)

    val = np.concatenate(val, axis=0)
    indicies = np.concatenate(indicies, axis=0)
    indptr = [non_zeros * i for i in range(N + 1)]

    C_sparse = sparse.csr_matrix((val, indicies, indptr), shape=[N, N])
    return C_sparse


def get_knn_Aff(C_sparse_normalized, k=10, mode="symmetric"):
    C_knn = kneighbors_graph(
        C_sparse_normalized, k, mode="connectivity", include_self=False, n_jobs=10
    )
    if mode == "symmetric":
        Aff_knn = 0.5 * (C_knn + C_knn.T)
    elif mode == "reciprocal":
        Aff_knn = C_knn @ C_knn.T
    else:
        raise Exception("Mode must be 'symmetric' or 'reciprocal'")
    return Aff_knn


def main():
    config = {
        "dataset": "coil-20",
        "gamma": 200.0,
        "lmbd": 0.9,
        "hid_dims": [1024, 1024, 1024],
        "out_dims": 1024,
        "total_iters": 500,
        "eval_iters": 200000,
        "lr": 1e-3,
        "lr_min": 0.0,
        "non_zeros": 1000,
        "n_neighbors": 3,
        "spectral_dim": 15,
        "affinity": "nearest_neighbor",
        "mean_subtraction": False,
        "knn_mode": "symmetric",
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    features = torch.load(f"dataset/embedding/resnet/{config['dataset']}_Feature.pt")
    labels = torch.load(
        f"dataset/embedding/resnet/{config['dataset']}_Label.pt"
    ).squeeze()

    config["batch_size"] = features.shape[0]
    config["chunk_size"] = features.shape[0]

    n_clusters = len(torch.unique(labels))
    config["num_subspaces"] = n_clusters

    trainer = Trainer(config, device)
    trainer.train(features, labels)
    trainer.evaluate(
        config["num_subspaces"],
        config["affinity"],
        config["knn_mode"],
    )


if __name__ == "__main__":
    main()
