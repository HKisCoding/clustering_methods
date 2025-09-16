import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoEncoder(nn.Module):
    """
    Convolutional Autoencoder implementation in PyTorch
    Converted from Keras/TensorFlow implementation
    """

    def __init__(self, input_shape=(1, 28, 28), filters=[32, 64, 128, 10]):
        """
        Args:
            input_shape: tuple, shape of input (channels, height, width)
            filters: list, number of filters for each layer [conv1, conv2, conv3, embedding_dim]
        """
        super(ConvAutoEncoder, self).__init__()

        self.input_shape = input_shape
        self.filters = filters
        self.channels, self.height, self.width = input_shape

        # Determine padding for conv3/deconv3 based on input size
        if self.height % 8 == 0:
            self.pad3 = 1  # 'same' padding for kernel size 3
        else:
            self.pad3 = 0  # 'valid' padding

        # Encoder layers
        self.conv1 = nn.Conv2d(
            self.channels, filters[0], kernel_size=5, stride=2, padding=2
        )  # 'same' padding
        self.conv2 = nn.Conv2d(
            filters[0], filters[1], kernel_size=5, stride=2, padding=2
        )  # 'same' padding
        self.conv3 = nn.Conv2d(
            filters[1], filters[2], kernel_size=3, stride=2, padding=self.pad3
        )

        # Calculate the flattened size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            self.conv_output_size = x.numel()
            self.feature_map_shape = x.shape[1:]  # (channels, height, width)

        # Embedding layers
        self.embedding = nn.Linear(self.conv_output_size, filters[3])
        self.decoder_input = nn.Linear(filters[3], self.conv_output_size)

        # Decoder layers
        self.deconv3 = nn.ConvTranspose2d(
            filters[2], filters[1], kernel_size=3, stride=2, padding=self.pad3
        )
        self.deconv2 = nn.ConvTranspose2d(
            filters[1], filters[0], kernel_size=5, stride=2, padding=2
        )
        self.deconv1 = nn.ConvTranspose2d(
            filters[0], self.channels, kernel_size=5, stride=2, padding=2
        )

        # For extracting features
        self.feature_extractor = None

    def encode(self, x):
        """Encoder forward pass"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        features = self.embedding(x)
        return features

    def decode(self, features):
        """Decoder forward pass"""
        x = F.relu(self.decoder_input(features))
        x = x.view(x.size(0), *self.feature_map_shape)  # Reshape to feature map shape

        # Handle output padding for transposed convolutions to match original size
        if self.pad3 == 1:  # 'same' case
            x = F.relu(
                self.deconv3(
                    x,
                    output_size=(
                        x.size(0),
                        self.filters[1],
                        self.feature_map_shape[1] * 2,
                        self.feature_map_shape[2] * 2,
                    ),
                )
            )
        else:  # 'valid' case
            x = F.relu(self.deconv3(x))

        x = F.relu(
            self.deconv2(
                x,
                output_size=(
                    x.size(0),
                    self.filters[0],
                    self.height // 2,
                    self.width // 2,
                ),
            )
        )
        x = self.deconv1(
            x, output_size=(x.size(0), self.channels, self.height, self.width)
        )

        return x

    def forward(self, x):
        """Full autoencoder forward pass"""
        features = self.encode(x)
        reconstructed = self.decode(features)
        return reconstructed

    def get_features(self, x):
        """Extract features from the embedding layer"""
        return self.encode(x)

    def summary(self):
        """Print model summary similar to Keras"""
        print("ConvAutoEncoder Model Summary:")
        print("=" * 50)
        print(f"Input shape: {self.input_shape}")
        print(f"Filters: {self.filters}")
        print(f"Conv3 padding: {'same' if self.pad3 == 1 else 'valid'}")
        print("=" * 50)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 50)

        # Print layer information
        print("\nLayer Information:")
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                print(f"{name}: {module}")


def CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
    """
    Create a Convolutional Autoencoder model

    Args:
        input_shape: tuple, input shape in format (height, width, channels) - Keras format
        filters: list, filter configuration [conv1, conv2, conv3, embedding_dim]

    Returns:
        ConvAutoEncoder model
    """
    # Convert from Keras format (H, W, C) to PyTorch format (C, H, W)
    if len(input_shape) == 3:
        pytorch_shape = (input_shape[2], input_shape[0], input_shape[1])
    else:
        pytorch_shape = input_shape

    model = ConvAutoEncoder(input_shape=pytorch_shape, filters=filters)
    model.summary()
    return model


if __name__ == "__main__":
    import argparse
    import os
    from time import time

    # Setting the hyper parameters
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--dataset", default="usps", choices=["mnist", "usps"])
    parser.add_argument("--n_clusters", default=10, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--save_dir", default="results/temp", type=str)
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str
    )
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load dataset
    from datasets import load_mnist, load_usps

    if args.dataset == "mnist":
        x, y = load_mnist()
    elif args.dataset == "usps":
        x, y = load_usps("data/usps")

    # Convert to PyTorch tensors and adjust shape
    # Assuming x comes in shape (N, H, W, C) from Keras loader
    if len(x.shape) == 4:
        x = torch.FloatTensor(x).permute(0, 3, 1, 2)  # Convert to (N, C, H, W)
    else:
        x = torch.FloatTensor(x).unsqueeze(1)  # Add channel dimension

    device = torch.device(args.device)
    x = x.to(device)

    # Define the model
    input_shape = x.shape[1:]  # (C, H, W)
    model = ConvAutoEncoder(input_shape=input_shape, filters=[32, 64, 128, 10])
    model = model.to(device)
    model.summary()

    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    t0 = time()

    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0

        # Create data loader
        dataset = torch.utils.data.TensorDataset(
            x, x
        )  # Input and target are the same for autoencoder
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True
        )

        for batch_x, batch_target in dataloader:
            optimizer.zero_grad()

            # Forward pass
            output = model(batch_x)
            loss = criterion(output, batch_target)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{args.epochs}], Loss: {avg_loss:.6f}")

    print("Training time: ", time() - t0)

    # Save the model
    torch.save(
        model.state_dict(),
        f"{args.save_dir}/{args.dataset}-pretrain-model-{args.epochs}.pth",
    )
    torch.save(
        model, f"{args.save_dir}/{args.dataset}-pretrain-full-model-{args.epochs}.pth"
    )

    # Extract features
    model.eval()
    with torch.no_grad():
        features = model.get_features(x)
        features = features.cpu().numpy()

    print("feature shape=", features.shape)

    # Use features for clustering
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=args.n_clusters)

    features = np.reshape(features, newshape=(features.shape[0], -1))
    pred = km.fit_predict(features)

    # Note: metrics import might need adjustment based on project structure
    try:
        from metrics import acc, ari, nmi

        y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        print("acc=", acc(y_np, pred), "nmi=", nmi(y_np, pred), "ari=", ari(y_np, pred))
    except ImportError:
        print("Metrics module not found. Skipping evaluation.")
        print(f"Predicted clusters: {np.unique(pred)}")
