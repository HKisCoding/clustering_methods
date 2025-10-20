import pandas as pd
import torch

from common import run_evaluate
from DEKM.torch_implement.trainer import DEKMDenseTrainer


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
        },
        "hidden_units": 10,
        "pretrain_epochs": 200,
        "pretrain_batch_size": 2000,
        "batch_size": 2000,
        "update_interval": 10,
    }
    DATASET_NAME = "USPS"

    features = torch.load(
        config["dataset"][DATASET_NAME]["feature_path"], map_location="cpu"
    )
    labels = torch.load(
        config["dataset"][DATASET_NAME]["label_path"], map_location="cpu"
    ).squeeze()

    n_clusters = len(torch.unique(labels))
    config["n_clusters"] = n_clusters
    config["input_shape"] = features.shape[1]
    eval_results = []
    for _ in range(5):
        trainer = DEKMDenseTrainer(
            input_shape=config["input_shape"],
            hidden_units=config["hidden_units"],
            n_clusters=config["n_clusters"],
            pretrain_epochs=config["pretrain_epochs"],
            pretrain_batch_size=config["pretrain_batch_size"],
            batch_size=config["batch_size"],
            update_interval=config["update_interval"],
        )

        trainer.pretrain(features)
        trainer.train(features, labels)

        assignment = trainer.predict_clusters(features)
        results = run_evaluate(assignment, labels.cpu().numpy(), n_clusters)
        eval_results.append(results)
    pd.DataFrame(eval_results).to_csv(f"output/DEKM/{DATASET_NAME}.csv", index=False)


if __name__ == "__main__":
    main()
