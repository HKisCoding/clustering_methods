import torch
from net.model import Model

from common import run_evaluate


class Trainer:
    def __init__(self, config, device, model):
        self.config = config
        self.device = device
        self.model = Model(
            mp_units=config.get("mp_units", [16]),
            mp_act=config.get("mp_act", "ELU"),
            in_channels=config.get("in_channels", 16),
            n_clusters=config.get("n_clusters", 16),
            mlp_units=config.get("mlp_units", []),
            mlp_act=config.get("mlp_act", "Identity"),
        )
        self.lr = config.get("lr", 1e-2)
        self.epochs = config.get("epochs", 10000)
        self.patience = config.get("patience", 50)

        self.n_clusters = config.get("n_clusters", 16)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Training state
        self.current_patience = self.patience

    def train(
        self, X: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()

        """Main training loop with early stopping"""
        for epoch in range(1, self.epochs + 1):
            # Training step
            # Forward pass
            _, mc_loss, o_loss = self.model(X, edge_index, edge_weight)

            # Combined loss
            loss = mc_loss + o_loss

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Print progress
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

    def evaluate(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Evaluation step"""
        self.model.eval()
        assigment, _, _ = self.model(X, edge_index, edge_weight)
        return run_evaluate(assigment, labels, self.n_clusters)
