import ptsdae.model as ae
import torch
from ptdec.dec import DEC
from ptdec.model import predict, train
from ptsdae.sdae import StackedDenoisingAutoEncoder
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, random_split

from common import run_evaluate


def main():
    config = {
        "dataset": "coil-20",
        "pretrain_epochs": 300,
        "finetune_epochs": 500,
        "batch_size": 256,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    features = torch.load(f"data/{config['dataset']}/features.pt")
    labels = torch.load(f"data/{config['dataset']}/labels.pt")

    train_size = int(0.9 * len(features))
    valid_size = len(features) - train_size
    dataset = TensorDataset(features, labels)
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    autoencoder = StackedDenoisingAutoEncoder(
        [28 * 28, 500, 500, 2000, 10], final_activation=None
    )
    print("Pretraining stage.")
    ae.pretrain(
        train_dataset,
        autoencoder,
        cuda=True,
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
        cuda=True,
        validation=valid_dataset,
        epochs=config["finetune_epochs"],
        batch_size=config["batch_size"],
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        corruption=0.2,
        update_callback=None,
    )
    print("DEC stage.")
    model = DEC(cluster_number=10, hidden_dimension=10, encoder=autoencoder.encoder)
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
    run_evaluate(actual, predicted, n_clusters=10)


if __name__ == "__main__":
    main()
