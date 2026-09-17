"""
End-to-end federated training simulation across N virtual hospital clients.

Run this script directly to produce backend/saved_models/global_model.pth,
which app.py loads to serve predictions:

    cd backend
    python -m federated.simulate_training

-------------------------------------------------------------------------
IMPORTANT -- ABOUT THE DATA
-------------------------------------------------------------------------
This script ships with a SYNTHETIC data generator so the whole pipeline
runs out of the box with no downloads. It proves the federated-learning
mechanics work end to end, but a model trained on random noise will not
give medically meaningful predictions.

For a real mini-project submission, replace `build_synthetic_client_data()`
with `build_real_client_data()` (stub provided below) pointing at the
HAM10000 dataset (https://doi.org/10.7910/DVN/DBW86T) or any labeled skin
lesion image folder. Split the images across N folders (one per simulated
hospital) to mimic non-IID, siloed hospital data -- that's what makes it
"federated" rather than plain centralized training.
-------------------------------------------------------------------------
"""

import os
import sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import CLASS_NAMES
from federated.fed_client import FLClient
from federated.fed_server import FLServer

NUM_CLIENTS = 4          # e.g. 4 simulated hospitals/clinics
SAMPLES_PER_CLIENT = 60  # small, since this is a CPU demo
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
IMAGE_SIZE = 64


class SyntheticSkinDataset(Dataset):
    """Placeholder dataset: random images + random (but fixed-seeded,
    class-biased) labels so different clients have different label
    distributions, mimicking real-world non-IID hospital data."""

    def __init__(self, num_samples: int, client_seed: int, num_classes: int = len(CLASS_NAMES)):
        g = torch.Generator().manual_seed(client_seed)
        self.images = torch.rand(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g)
        # bias each client toward a couple of "locally common" classes
        bias_classes = torch.randperm(num_classes, generator=g)[:2]
        labels = []
        for _ in range(num_samples):
            if torch.rand(1, generator=g).item() < 0.6:
                labels.append(bias_classes[torch.randint(0, 2, (1,), generator=g).item()].item())
            else:
                labels.append(torch.randint(0, num_classes, (1,), generator=g).item())
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def build_synthetic_client_data():
    return [
        SyntheticSkinDataset(SAMPLES_PER_CLIENT, client_seed=100 + i)
        for i in range(NUM_CLIENTS)
    ]


def build_real_client_data(root_dir: str):
    """
    STUB for real data. Expected layout:

        root_dir/
          hospital_1/<class_name>/*.jpg
          hospital_2/<class_name>/*.jpg
          ...

    Use torchvision.datasets.ImageFolder per hospital_N subfolder so each
    client gets its own DataLoader-compatible dataset, e.g.:

        from torchvision.datasets import ImageFolder
        tfm = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        return [ImageFolder(os.path.join(root_dir, d), transform=tfm)
                for d in sorted(os.listdir(root_dir))]
    """
    raise NotImplementedError("Point this at your real, per-hospital image folders.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    client_datasets = build_synthetic_client_data()
    clients = [
        FLClient(client_id=f"hospital_{i+1}", dataset=ds, device=device)
        for i, ds in enumerate(client_datasets)
    ]

    server = FLServer(device=device)

    print(f"\nStarting federated training: {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds\n")
    for round_num in range(1, NUM_ROUNDS + 1):
        losses = server.run_round(clients, local_epochs=LOCAL_EPOCHS)
        loss_str = ", ".join(f"{cid}={loss:.4f}" for cid, loss in losses.items())
        print(f"Round {round_num}/{NUM_ROUNDS} — local losses: {loss_str}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "global_model.pth")
    server.save(out_path)
    print(f"\nSaved aggregated global model to {out_path}")


if __name__ == "__main__":
    main()
