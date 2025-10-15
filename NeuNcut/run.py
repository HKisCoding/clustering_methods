import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from loss import ncut_loss
from models import MLP
from tqdm import tqdm

from common import run_evaluate
from NeuNcut.utils import create_affinity_matrix, p_normalize


def load_data(dataset):
    if dataset == "MNIST":
        features = torch.load("dataset/embedding/auto_encoder/mnist_raw_Feature.pt")
        labels = torch.load(
            "dataset/embedding/auto_encoder/mnist_raw_Label.pt"
        ).squeeze()
    elif dataset == "coil-20":
        features = torch.load("dataset/embedding/resnet/coil-20_Feature.pt")
        labels = torch.load("dataset/embedding/resnet/coil-20_Label.pt").squeeze()
    elif dataset == "MSRC-v2":
        features = torch.load("dataset/embedding/resnet/MSRC-v2_Feature.pt")
        labels = torch.load("dataset/embedding/resnet/MSRC-v2_Label.pt").squeeze()

    return features, labels


def run():
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
            "Caltech_101": {
                "feature_path": "dataset/embedding/resnet/Caltech_101_Feature.pt",
                "label_path": "dataset/embedding/resnet/Caltech_101_Label.pt",
            },
        },
        "seed": 0,
        "hid_dims": [512, 512],
        "epo": 300,
        "lr": 5e-3,
        "wd": 1e-4,
        "gamma": 80,
        "sigma": 3.0,
        "ctn": False,
        "step": 50,
        "p_scale": 1.1,
        "g_max": 80,
        "bs": 2000,
    }
    DATASET_NAME = "USPS"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_data = torch.load(
        config["dataset"][DATASET_NAME]["feature_path"], map_location="cpu"
    )
    full_labels = torch.load(
        config["dataset"][DATASET_NAME]["label_path"], map_location="cpu"
    ).squeeze()

    n_clusters = len(torch.unique(full_labels))
    config["n_classes"] = n_clusters
    # config["bs"] = full_data.shape[0]

    data = p_normalize(full_data)
    labels = full_labels
    eval_results = []
    for _ in range(5):
        # NeuNcut instance
        cls_head = MLP(data.shape[1], config["hid_dims"], config["n_classes"]).to(
            device
        )

        n_iter_per_epoch = full_data.shape[0] // config["bs"]

        optimizer = optim.Adam(
            cls_head.parameters(), lr=config["lr"], weight_decay=config["wd"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epo"])

        pbar = tqdm(range(config["epo"]))
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            randidx = torch.randperm(full_data.shape[0])
            cls_head.train()
            losses = []
            for i in range(n_iter_per_epoch):
                batch_idx = randidx[i * config["bs"] : (i + 1) * config["bs"]]
                batch = data[batch_idx].contiguous().to(device)

                # Compute euclidean affinities
                W = create_affinity_matrix(batch, 10, 20, device)

                # Get soft predictions
                P = torch.softmax(cls_head(batch), dim=1)

                # Compute NeuNcut loss
                spectral_loss, orth_reg = ncut_loss(W, P)
                loss = spectral_loss + 0.5 * config["gamma"] * orth_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            scheduler.step()

            with torch.no_grad():
                cls_head.eval()
                pred = []
                for i in range(full_data.shape[0] // config["bs"]):
                    batch = data[i * config["bs"] : (i + 1) * config["bs"]].to(device)
                    logits = torch.softmax(cls_head(batch), dim=1)
                    batch_pred = torch.argmax(logits, dim=1)
                    pred.extend(list(batch_pred.cpu().data.numpy()))
                pred = np.array(pred)
                # results = run_evaluate(pred, labels, config["n_classes"])
                pbar.set_postfix(loss="{:3.4f}".format(np.mean(losses)))

        print("evaluating on {}-full...".format(config["dataset"]))
        full_data = p_normalize(full_data).to(device)
        pred = []
        for i in range(full_data.shape[0] // config["bs"]):
            batch = full_data[i * config["bs"] : (i + 1) * config["bs"]].to(device)
            logits = cls_head(batch)
            temp_pred = torch.argmax(logits, dim=1).cpu().data.numpy()
            pred.extend(list(temp_pred))
        pred = np.array(pred)
        results = run_evaluate(pred, labels.cpu().numpy(), config["n_classes"])
        eval_results.append(results)

        print(results)
        pd.DataFrame(eval_results).to_csv(
            f"output/neuncut/{DATASET_NAME}.csv", index=False
        )


if __name__ == "__main__":
    run()
