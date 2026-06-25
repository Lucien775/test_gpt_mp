import csv
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class ErrorAnalyzer:
    """
    Compares inference behavior across any set of models (different precisions,
    mixed-precision configs, architectures, etc.).

    The analyzer is model-agnostic: it receives already-built and weight-loaded
    models, and only handles data loading, forward passes, and result saving.

    Args:
        models:     dict[name -> nn.Module], already instantiated, weights loaded, on device
        data_path:  path to the raw text file (stoi/itos come from the checkpoint,
                    passed separately)
        stoi:       char-to-index mapping from the checkpoint
        itos:       index-to-char mapping from the checkpoint
        block_size: sequence length (from the checkpoint config)
        batch_size: number of sequences per batch
        device:     torch device
    """

    def __init__(
        self,
        models: dict[str, nn.Module],
        data_path: str,
        stoi: dict,
        itos: dict,
        block_size: int,
        batch_size: int,
        device: torch.device,
    ):
        self.models = models
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: "".join(itos[i] for i in l)

        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str):
        """Return a random batch from 'train' or 'val' split."""
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i: i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1: i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)

    def compare(self, num_batches: int = 500, split: str = "val") -> dict:
        """
        Run forward passes for every model and collect losses.

        Returns:
            {
                'model_name': {'losses': [...], 'avg_loss': float},
                ...
            }
        """
        results = {name: {"losses": []} for name in self.models}

        for _ in tqdm(range(num_batches), desc="Evaluating"):
            xb, yb = self.get_batch(split)
            for name, model in self.models.items():
                with torch.no_grad():
                    _, loss = model(xb, yb)
                    results[name]["losses"].append(loss.item())

        for name in self.models:
            results[name]["avg_loss"] = float(np.mean(results[name]["losses"]))

        return results

    def save_results(self, results: dict, output_path: str):
        """Save compare() results to a CSV file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "avg_loss"])
            for name, metrics in results.items():
                writer.writerow([name, f"{metrics['avg_loss']:.6f}"])
        print(f"Results saved to {output_path}")