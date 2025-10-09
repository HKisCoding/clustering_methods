import numpy as np
import pandas as pd
import torch
from model import GraphEncoder
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn, optim
from tqdm import tqdm

from common import run_evaluate


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
        },
        "layers": [128, 64, 128],
        "beta": 0.01,
        "rho": 0.5,
        "lr": 0.01,
        "epoch": 200,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    DATASET_NAME = "USPS"

    features = torch.load(
        config["dataset"][DATASET_NAME]["feature_path"], map_location="cpu"
    )
    labels = torch.load(
        config["dataset"][DATASET_NAME]["label_path"], map_location="cpu"
    ).squeeze()

    n_clusters = len(torch.unique(labels))

    X = features.numpy()
    Y = labels.numpy()

    # Obtain Similarity matrix
    S = cosine_similarity(X, X)
    # Normalize cosine similarity from [-1, 1] to [0, 1]
    S = (S + 1) / 2

    D = np.diag(1.0 / np.sqrt(S.sum(axis=1)))
    X_train = torch.tensor(D.dot(S).dot(D)).float().to(config["device"])

    layers = [len(X_train)] + config["layers"] + [len(X_train)]

    model = GraphEncoder(layers, n_clusters).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    pbar = tqdm(range(config["epoch"]))
    eval_results = []
    for _ in range(10):
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
        eval_results.append(results)

    pd.DataFrame(eval_results).to_csv(
        f"output/graphencoder/{DATASET_NAME}.csv", index=False
    )


if __name__ == "__main__":
    main()
