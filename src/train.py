"""
Training script for turbofan engine RUL prediction model.

Usage:
    python -m src.train --epochs 50 --batch-size 256
"""

import argparse
import time
import json
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import (
    CMAPSSDataLoader,
    check_data_available,
    print_download_instructions,
    FEATURE_COLUMNS,
)
from src.model import create_model, count_parameters


def create_dataloaders(
    loader: CMAPSSDataLoader,
    window_size: int = 30,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Uses 70/30 engine split: train_ids for training, demo_ids for validation.

    Args:
        loader: CMAPSSDataLoader with loaded data
        window_size: Sequence length for LSTM
        batch_size: Batch size
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get training sequences (from train_ids - 70% of engines)
    X_train, y_train = loader.get_training_sequences(window_size=window_size)

    # Get validation sequences (from demo_ids - 30% of engines)
    X_val, y_val = loader.get_validation_sequences(window_size=window_size)

    print(f"Training sequences: {len(X_train)} (from {len(loader.train_ids)} engines)")
    print(f"Validation sequences: {len(X_val)} (from {len(loader.demo_ids)} engines)")
    print(f"Sequence shape: {X_train.shape[1:]} (window_size, n_features)")
    print(f"Train RUL range: {y_train.min():.0f} - {y_train.max():.0f}")
    print(f"Val RUL range: {y_val.min():.0f} - {y_val.max():.0f}")

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train).unsqueeze(1)
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val).unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        rul_pred = model(X_batch)
        loss = criterion(rul_pred, y_batch)

        loss.backward()

        # Gradient clipping for LSTM stability
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        total_samples += len(X_batch)

    return {
        "loss": total_loss / total_samples,
        "rmse": np.sqrt(total_loss / total_samples),
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            rul_pred = model(X_batch)
            loss = criterion(rul_pred, y_batch)

            total_loss += loss.item() * len(X_batch)
            total_samples += len(X_batch)

            all_preds.extend(rul_pred.cpu().numpy().flatten())
            all_targets.extend(y_batch.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute metrics
    mse = total_loss / total_samples
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))

    # Score function (common for RUL prediction)
    # Penalizes late predictions more than early ones
    diff = all_preds - all_targets
    score = np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1))

    return {
        "loss": mse,
        "rmse": rmse,
        "mae": mae,
        "score": score,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    learning_rate: float = 0.001,
    save_path: str = "models/rul_model.pt",
) -> Dict:
    """
    Train the model.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Number of epochs
        learning_rate: Learning rate
        save_path: Path to save best model

    Returns:
        Training history dictionary
    """
    model = model.to(device)

    # Loss function (MSE for regression)
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {
        "train_loss": [],
        "train_rmse": [],
        "val_loss": [],
        "val_rmse": [],
        "val_mae": [],
        "val_score": [],
    }

    best_rmse = float('inf')

    print(f"\nTraining on {device}")
    print(f"Model parameters: {count_parameters(model):,}")
    print("-" * 80)

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler based on validation loss
        scheduler.step(val_metrics["loss"])

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_rmse"].append(train_metrics["rmse"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_score"].append(val_metrics["score"])

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train RMSE: {train_metrics['rmse']:.2f} | "
            f"Val RMSE: {val_metrics['rmse']:.2f} | "
            f"Val MAE: {val_metrics['mae']:.2f} | "
            f"Score: {val_metrics['score']:.0f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, save_path)
            print(f"  -> Saved best model (RMSE: {best_rmse:.2f})")

        # Save latest model
        latest_path = save_path.replace(".pt", "_latest.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
        }, latest_path)

    print("-" * 80)
    print(f"Best validation RMSE: {best_rmse:.2f}")
    print(f"Best model saved to: {save_path}")
    print(f"Latest model saved to: {latest_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train RUL prediction model")
    parser.add_argument("--data-dir", type=str, default="data/CMAPSSData",
                        help="Path to C-MAPSS data directory")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["FD001", "FD002", "FD003", "FD004"],
                        help="Dataset subsets to use")
    parser.add_argument("--model-type", type=str, default="lstm",
                        choices=["lstm", "cnn_lstm", "attention"],
                        help="Model architecture")
    parser.add_argument("--window-size", type=int, default=30,
                        help="Sequence window size")
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="LSTM hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout probability")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--demo-ratio", type=float, default=0.3,
                        help="Demo set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save-path", type=str, default="models/rul_model.pt",
                        help="Path to save model")

    args = parser.parse_args()

    # Check data
    if not check_data_available(args.data_dir):
        print("Dataset not found!")
        print_download_instructions()
        return

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\nLoading dataset...")
    loader = CMAPSSDataLoader(data_dir=args.data_dir, datasets=args.datasets)
    loader.load()
    loader.create_train_demo_split(demo_ratio=args.demo_ratio, seed=args.seed)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        loader,
        window_size=args.window_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Get number of features
    n_features = len(FEATURE_COLUMNS)
    print(f"Number of input features: {n_features}")

    # Create model
    model = create_model(
        model_type=args.model_type,
        input_size=n_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        save_path=args.save_path,
    )

    # Save training info
    info_path = args.save_path.replace(".pt", "_info.json")
    with open(info_path, "w") as f:
        json.dump({
            "model_type": args.model_type,
            "window_size": args.window_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "datasets": args.datasets,
            "train_engines": len(loader.train_ids),
            "demo_engines": len(loader.demo_ids),
            "best_val_rmse": float(min(history["val_rmse"])),
            "best_val_mae": float(min(history["val_mae"])),
        }, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
