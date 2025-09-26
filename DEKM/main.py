import torch

from DEKM.torch_implement.trainer import DEKMDenseTrainer


def main():
    config = {"hidden_units": 10, "dataset": "coil-20"}

    features = torch.load(f"dataset/embedding/resnet/{config['dataset']}_Feature.pt")
    labels = torch.load(
        f"dataset/embedding/resnet/{config['dataset']}_Label.pt"
    ).squeeze()

    n_clusters = len(torch.unique(labels))
    config["n_clusters"] = n_clusters
    config["input_shape"] = features.shape[1]

    trainer = DEKMDenseTrainer(
        input_shape=config["input_shape"],
        hidden_units=config["hidden_units"],
        n_clusters=config["n_clusters"],
    )

    trainer.pretrain(features)
    trainer.train(features, labels)


if __name__ == "__main__":
    main()
