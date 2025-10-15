import pandas as pd
import ptdec.ae as ae
import torch
from ptdec.dec import DEC
from ptdec.model import predict, train
from ptdec.sdae import StackedDenoisingAutoEncoder
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, random_split

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
            "mnist": {
                "feature_path": "dataset/embedding/auto_encoder/mnist_raw_Feature.pt",
                "label_path": "dataset/embedding/auto_encoder/mnist_raw_Label.pt",
            },
            "Caltech_101": {
                "feature_path": "dataset/embedding/resnet/Caltech_101_Feature.pt",
                "label_path": "dataset/embedding/resnet/Caltech_101_Label.pt",
            },
        },
        "pretrain_epochs": 300,
        "finetune_epochs": 500,
        "batch_size": 256,
    }
    DATASET_NAME = "coil-20"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    features = torch.load(config["dataset"][DATASET_NAME]["feature_path"])
    labels = torch.load(config["dataset"][DATASET_NAME]["label_path"])

    train_size = int(0.9 * len(features))
    valid_size = len(features) - train_size
    dataset = TensorDataset(features, labels)
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    n_clusters = len(torch.unique(labels))
    dimensions = [features.shape[1], 500, 500, 2000, 10]
    eval_results = []
    for _ in range(5):
        autoencoder = StackedDenoisingAutoEncoder(dimensions, final_activation=None)
        print("Pretraining stage.")
        ae.pretrain(
            train_dataset,
            autoencoder,
            device=device,
            validation=valid_dataset,
            epochs=config["pretrain_epochs"],
            batch_size=config["batch_size"],
            optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
            scheduler=lambda x: StepLR(x, 100, gamma=0.1),
            corruption=0.2,
        )
        print("Training stage.")
        ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
        ae.train(
            train_dataset,
            autoencoder,
            device=device,
            validation=valid_dataset,
            epochs=config["finetune_epochs"],
            batch_size=config["batch_size"],
            optimizer=ae_optimizer,
            scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
            corruption=0.2,
            update_callback=None,
        )
        print("DEC stage.")
        model = DEC(
            cluster_number=n_clusters, hidden_dimension=10, encoder=autoencoder.encoder
        ).to(device)
        dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        train(
            dataset=train_dataset,
            model=model,
            epochs=100,
            batch_size=256,
            optimizer=dec_optimizer,
            stopping_delta=0.000001,
            device=device,
        )
        predicted, actual = predict(
            train_dataset, model, 1024, silent=True, return_actual=True, device=device
        )
        actual = actual.cpu().numpy()
        predicted = predicted.cpu().numpy()
        results = run_evaluate(predicted, labels.cpu().numpy(), config["n_classes"])
        eval_results.append(results)

        print(results)
        pd.DataFrame(eval_results).to_csv(f"output/dec/{DATASET_NAME}.csv", index=False)


if __name__ == "__main__":
    main()
