import os
import sys
from pathlib import Path
from tqdm import tqdm

from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import logging

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.path.join(sys.path[0], project_dir))
from src.model.models_architectures import MODEL_ARCHITECTURES
from src.data.make_dataset import *
from src.model.utils import get_timestamp, setup_logger, plot_losses


def train_model(X_train, y_train, X_val, y_val, model, lr=0.0001, batch_size=16, max_epochs=20, patience=2):
    logger = setup_logger()
    logger.info("Preparing training and validation datasets...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # print("X_train shape BEFORE tensor:", X_train.shape)
    # print("X_val shape BEFORE tensor:", X_val.shape)

    # Convert to CPU tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # shape: (B, 1, T)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # shape: (B, 1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    logger.info(f"X_train_tensor: {X_train_tensor.shape}")
    logger.info(f"y_train_tensor: {y_train_tensor.shape}")
    logger.info(f"X_val_tensor: {X_val_tensor.shape}")
    logger.info(f"y_val_tensor: {y_val_tensor.shape}")

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=batch_size, pin_memory=True
    )

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    os.makedirs("models", exist_ok=True)
    timestamp = get_timestamp()
    model_path = f"models/{timestamp}_best_model.pth"

    logger.info("Starting training...")

    for epoch in range(max_epochs):
        logger.info(f"Epoch {epoch+1}/{max_epochs} - Training...")
        model.train()
        epoch_loss = 0.0

        for xb, yb in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=True):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch {epoch+1} - Training loss: {avg_train_loss:.6f}")

        # Validation
        logger.info(f"Epoch {epoch+1} - Validating...")
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False):
                xb = xb.to(device)
                yb = yb.to(device)

                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logger.info(f"Epoch {epoch+1} - Validation loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            logger.info(f"Epoch {epoch+1} - New best model found! Saving to {model_path}")
            best_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, model_path)
        else:
            epochs_no_improve += 1
            logger.info(f"Epoch {epoch+1} - No improvement. Patience: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

    model.load_state_dict(torch.load(model_path))
    logger.info(f"Training complete. Best model loaded from {model_path}")

    return model, model_path, train_losses, val_losses, timestamp


if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test, norm_params = make_dataset2()
    model = MODEL_ARCHITECTURES['Seq2PointOneToOne']()
    trained_model, model_path, train_losses, val_losses, timestamp = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model=model,
        lr=0.01,
        batch_size=1024,
        max_epochs=300,
        patience=10)
    plot_losses(trained_model, model_path, train_losses, val_losses, timestamp, save=True)





