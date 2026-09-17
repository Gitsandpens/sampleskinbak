"""
Federated Learning server.

Implements FedAvg (McMahan et al., 2017): the global model is the
sample-weighted average of every participating client's local weights.
The server never sees raw client data -- only weight tensors.
"""

import copy
import torch

from models.cnn_model import SkinDiseaseCNN


class FLServer:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.global_model = SkinDiseaseCNN().to(device)

    def get_global_weights(self) -> dict:
        return copy.deepcopy(self.global_model.state_dict())

    def federated_average(self, client_updates: list[tuple[dict, int]]) -> dict:
        """
        client_updates: list of (state_dict, num_samples) from each client.
        Weighted by number of local samples, per the FedAvg algorithm.
        """
        total_samples = sum(n for _, n in client_updates)
        new_state = copy.deepcopy(client_updates[0][0])

        for key in new_state.keys():
            new_state[key] = torch.zeros_like(new_state[key], dtype=torch.float32)

        for state_dict, num_samples in client_updates:
            weight = num_samples / total_samples
            for key in new_state.keys():
                new_state[key] += state_dict[key].float() * weight

        return new_state

    def run_round(self, clients, local_epochs: int = 1) -> dict:
        """Run one federated round: broadcast -> local train -> aggregate."""
        global_weights = self.get_global_weights()
        client_updates = []
        round_losses = {}

        for client in clients:
            updated_weights, num_samples, avg_loss = client.local_train(global_weights, local_epochs)
            client_updates.append((updated_weights, num_samples))
            round_losses[client.client_id] = avg_loss

        new_global_weights = self.federated_average(client_updates)
        self.global_model.load_state_dict(new_global_weights)

        return round_losses

    def save(self, path: str):
        torch.save(self.global_model.state_dict(), path)

    def load(self, path: str):
        self.global_model.load_state_dict(torch.load(path, map_location=self.device))
