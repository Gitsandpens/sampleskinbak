"""
Federated Learning client.

Each client represents a hospital/clinic that holds its own private image
data shard. Data NEVER leaves the client -- only model weight updates are
sent to the server, which is the core privacy guarantee of federated
learning.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.cnn_model import SkinDiseaseCNN


class FLClient:
    def __init__(self, client_id: str, dataset, batch_size: int = 16, lr: float = 0.001, device: str = "cpu"):
        self.client_id = client_id
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.lr = lr
        self.device = device

    def local_train(self, global_state_dict: dict, local_epochs: int = 1) -> tuple[dict, int, float]:
        """
        Train locally starting from the current global model weights.

        Returns:
            (updated_state_dict, num_samples, avg_training_loss)
        """
        model = SkinDiseaseCNN().to(self.device)
        model.load_state_dict(copy.deepcopy(global_state_dict))
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        total_loss, num_batches = 0.0, 0
        for _epoch in range(local_epochs):
            for images, labels in self.loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return model.state_dict(), len(self.dataset), avg_loss
