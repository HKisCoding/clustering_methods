import numpy as np
import torch
import torch.optim as optim
from loss import ncut_loss
from models import MLP

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
        "dataset": "coil-20",
        "n_classes": 10,
        "N": 20000,
        "seed": 0,
        "hid_dims": [512, 512],
        "epo": 300,
        "bs": 1000,
        "lr": 5e-3,
        "wd": 1e-4,
        "gamma": 80,
        "sigma": 3.0,
        "ctn": False,
        "step": 50,
        "p_scale": 1.1,
        "g_max": 80,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_data, full_labels = load_data(config["dataset"])
    data = p_normalize(full_data)
    labels = full_labels

    # NeuNcut instance
    cls_head = MLP(data.shape[1], config["hid_dims"], config["n_classes"]).to(device)

    n_iter_per_epoch = config["N"] // config["bs"]

    optimizer = optim.Adam(
        cls_head.parameters(), lr=config["lr"], weight_decay=config["wd"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epo"])

    for epoch in range(config["epo"]):
        randidx = torch.randperm(config["N"])
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
            for i in range(config["N"] // config["bs"]):
                batch = data[i * config["bs"] : (i + 1) * config["bs"]].to(device)
                logits = torch.softmax(cls_head(batch), dim=1)
                batch_pred = torch.argmax(logits, dim=1)
                pred.extend(list(batch_pred.cpu().data.numpy()))
            pred = np.array(pred)
            results = run_evaluate(pred, labels, config["n_classes"])
            print(f"Epoch {epoch + 1}: {results} | Loss: {np.mean(losses)}")

    print("evaluating on {}-full...".format(config["dataset"]))
    full_data = p_normalize(torch.from_numpy(full_data).float()).to(device)
    pred = []
    for i in range(full_data.shape[0] // 10000):
        batch = full_data[i * 10000 : (i + 1) * 10000].to(device)
        logits = cls_head(batch)
        temp_pred = torch.argmax(logits, dim=1).cpu().data.numpy()
        pred.extend(list(temp_pred))
    pred = np.array(pred)
    results = run_evaluate(pred, labels, config["n_classes"])

    print(results)


if __name__ == "__main__":
    run()
