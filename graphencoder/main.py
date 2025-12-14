import numpy as np
import pandas as pd
import torch
from model import GraphEncoder
from sklearn.metrics.pairwise import cosine_similarity
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from common import run_evaluate


def main():
    config = {
        "dataset": {
            "Caltech_101": {
                "feature_path": "dataset/embedding/resnet/Caltech_101_Feature.pt",
                "label_path": "dataset/embedding/resnet/Caltech_101_Label.pt",
            },
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
        },
        "layers": [128, 64, 128],
        "batch_size": 2000,
        "beta": 0.01,
        "rho": 0.5,
        "lr": 0.01,
        "epoch": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    DATASET_NAME = "mnist"

    features = torch.load(
        config["dataset"][DATASET_NAME]["feature_path"], map_location="cpu"
    )
    labels = torch.load(
        config["dataset"][DATASET_NAME]["label_path"], map_location="cpu"
    ).squeeze()

    n_clusters = len(torch.unique(labels))

    X = features.numpy()
    Y = labels.numpy()

    # Create dataloader from X and Y
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)

    # Construct layers as [batch_size, config["layers"], batch_size]
    # layers = [features.shape[0]] + config["layers"] + [features.shape[0]]
    layers = [config["batch_size"]] + config["layers"] + [config["batch_size"]]


    eval_results = []
    for _ in range(5):
        model = GraphEncoder(layers, n_clusters).to(config["device"])
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

        pbar = tqdm(range(config["epoch"]))

        for epoch in pbar:
            epoch_loss = 0.0
            for batch_X, batch_Y in dataloader:
                pbar.set_description(f"Epoch {epoch}")

                batch_X = batch_X.to(config["device"])

                # Construct similarity matrix S for this batch
                S = cosine_similarity(batch_X, batch_X)
                # Normalize cosine similarity from [-1, 1] to [0, 1]
                S = (S + 1) / 2

                D = np.diag(1.0 / np.sqrt(S.sum(axis=1)))
                batch_X_train = torch.tensor(D.dot(S).dot(D)).float().to(config["device"])

                optimizer.zero_grad()
                batch_X_hat = model(batch_X_train)
                loss = model.loss(batch_X_hat, batch_X_train, config["beta"], config["rho"])

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            pbar.set_postfix(loss="{:.3f}".format(epoch_loss / len(dataloader)))
            pbar.update()

        ACC_pred = 0.0
        NMI_pred = 0.0
        PURITY_pred = 0.0
        for batch_X, batch_Y in dataloader:
            batch_Y = batch_Y.to(config["device"])
            S = cosine_similarity(batch_X, batch_X)
            # Normalize cosine similarity from [-1, 1] to [0, 1]
            S = (S + 1) / 2

            D = np.diag(1.0 / np.sqrt(S.sum(axis=1)))
            batch_X_train = torch.tensor(D.dot(S).dot(D)).float().to(config["device"])

            with torch.no_grad():
                batch_X_hat = model(batch_X_train)
                cluster = model.get_cluster()
                results = run_evaluate(cluster, batch_Y.detach().cpu().numpy(), n_clusters)
                ACC_pred += results["ACC"]
                NMI_pred += results["NMI"]
                PURITY_pred += results["PURITY"]

        eval_results.append({
            "ACC": ACC_pred / len(dataloader),
            "NMI": NMI_pred / len(dataloader),
            "PURITY": PURITY_pred / len(dataloader),
        })



    pd.DataFrame(eval_results).to_csv(
        f"output/graphencoder/{DATASET_NAME}.csv", index=False
    )


if __name__ == "__main__":
    main()
