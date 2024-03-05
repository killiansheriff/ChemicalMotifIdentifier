import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class ModelInference:
    def __init__(self, model, dataset, batch_size=32):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = model.to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_loader = self._create_data_loader()

    def _create_data_loader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

    def predict(self):
        results = []
        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Inference"):
                inputs = batch.to(self.device)
                outputs = self.model(inputs)
                results.append(outputs.cpu().numpy())

        return np.concatenate(results, axis=0)


class ModelTrainer:
    def __init__(
        self,
        root,
        model,
        train_dataset,
        val_dataset,
        loss_fn,
        optimizer,
        checkpoint_path=None,
    ):
        """Initializes the PyTorchTrainer class.

        Args:
            model (_type_): _description_
            train_dataset (_type_): _description_
            val_dataset (_type_): _description_
            loss_fn (_type_): _description_
            optimizer (_type_): _description_
            checkpoint_path (_type_, optional): _description_. Defaults to None.
        """
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.root = root

        self.checkpoint_path = checkpoint_path
        self.train_loss = None
        self.val_loss = None

    def train(self, num_epochs, batch_size=32, verbose=True, start_epoch=0):
        if self.checkpoint_path is not None:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            if verbose:
                print(f"Loaded checkpoint from {self.checkpoint_path}")
        else:
            best_val_loss = float("inf")

        #!!! ADD NUM WORKERS
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs"):
            # Training loop
            train_loss = 0.0
            self.model.train()
            for x in tqdm(train_loader, mininterval=60):
                x = x.to(self.device)
                y_true = x.label
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y_true)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * x.size(0)
                # print(f'{train_loss/len(self.train_dataset)}')
            train_loss /= len(self.train_dataset)
            self.train_loss = train_loss

            # Validation loop
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(self.device)
                    y_true = x.label

                    y_pred = self.model(x)
                    loss = self.loss_fn(y_pred, y_true)
                    val_loss += loss.item() * x.size(0)
                val_loss /= len(self.val_dataset)
                self.val_loss = val_loss

            # Print progress and save checkpoint if val_loss improves
            if verbose:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.10f}, Val Loss: {val_loss:.10f}"
                )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                }

                os.makedirs(self.root + "checkpoints/", exist_ok=True)
                torch.save(
                    checkpoint,
                    self.root + f"checkpoints/chkpt_valloss{val_loss:.10f}.pt",
                )
                if verbose:
                    print(f"Saved checkpoint to {self.checkpoint_path}")
