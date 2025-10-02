import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn, optim
from tqdm import tqdm

from common import run_evaluate
from graphencoder.model import GraphEncoder


def main():
    config = {
        "dataset": "coil-20",
        "layers": [128, 64, 128],
        "beta": 0.01,
        "rho": 0.5,
        "lr": 0.01,
        "epoch": 200,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    features = torch.load(
        f"dataset/embedding/resnet/{config['dataset']}_Feature.pt", map_location="cpu"
    )
    labels = torch.load(
        f"dataset/embedding/resnet/{config['dataset']}_Label.pt", map_location="cpu"
    ).squeeze()

    n_clusters = len(torch.unique(labels))

    X = features.numpy()
    Y = labels.numpy()

    # Obtain Similarity matrix
    S = cosine_similarity(X, X)

    D = np.diag(1.0 / np.sqrt(S.sum(axis=1)))
    X_train = torch.tensor(D.dot(S).dot(D)).float().to(config["device"])

    layers = [len(X_train)] + config["layers"] + [len(X_train)]

    model = GraphEncoder(layers, n_clusters).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    pbar = tqdm(range(config["epoch"]))
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")
        optimizer.zero_grad()
        X_hat = model(X_train)
        loss = model.loss(X_hat, X_train, config["beta"], config["rho"])

        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss="{:.3f}".format(loss))
        pbar.update()

    cluster = model.get_cluster()
    results = run_evaluate(cluster, Y, n_clusters)
    print(results)


if __name__ == "__main__":
    main()
