import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.path.join(sys.path[0], project_dir))
from src.model.models_architectures import MODEL_ARCHITECTURES
from src.common.data_utils import make_dataset
from src.common.other_utils import get_timestamp, setup_logger, plot_losses, load_yaml_config


def freeze_layers(model):
    logger = setup_logger()

    # Freeze Feature Extraction Module
    logger.info("Freezing Feature Extraction Module: spatial_small, spatial_large, bigru1, bigru2, bigru3")
    for module in [model.spatial_small, model.spatial_large,
                   model.bigru1, model.bigru2, model.bigru3]:
        for param in module.parameters():
            param.requires_grad = False

    # Freeze nothing inside CBAM (Feature Adaptive Module trains)
    logger.info("Unfreezing Feature Adaptive Module (CBAM)")
    for param in model.cbam.parameters():
        param.requires_grad = True

    # Output Module trains
    logger.info("Unfreezing Output Module: fc1, output_linear, output_sigmoid")
    for module in [model.fc1, model.output_linear, model.output_sigmoid]:
        for param in module.parameters():
            param.requires_grad = True

    return model


def train_at_ht_model(X_train, y_train, X_val, y_val, model, lr=0.0001, batch_size=16, max_epochs=20, patience=2):
    logger = setup_logger()
    logger.info("Preparing training and validation datasets for CASE 2 (AT)...")

    config_paths = load_yaml_config("configs/paths_drive.yaml")
    paths = config_paths["output"]

    models_dir = paths["models"]
    losses_dir = paths["results"]["losses"]

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(losses_dir, exist_ok=True)

    logger.info(f"Model directory: {models_dir}")
    logger.info(f"Loss plots directory: {losses_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=batch_size, pin_memory=True
    )

    model = model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    trainable_layers = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen_layers = [n for n, p in model.named_parameters() if not p.requires_grad]
    print("Trainable:", trainable_layers)
    print("Frozen:", frozen_layers)

    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    os.makedirs("models", exist_ok=True)
    timestamp = get_timestamp()
    model_path = os.path.join(models_dir, f"{timestamp}_best_model.pth")
    logger.info(f"Best model will be saved to: {model_path}")

    logger.info("Starting CASE 2 training (AT)...")

    for epoch in range(max_epochs):
        logger.info(f"Epoch {epoch+1}/{max_epochs} - Training...")

        model.train()
        epoch_loss = 0.0

        for xb, yb in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb.unsqueeze(1))
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch {epoch+1} - Training loss: {avg_train_loss:.6f}")

        # Validation
        logger.info(f"Epoch {epoch + 1} - Validating...")
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb.unsqueeze(1))
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
            if epochs_no_improve > patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

    model.load_state_dict(torch.load(model_path))
    logger.info(f"CASE 2 training complete. Best model loaded from {model_path}")

    return model, model_path, train_losses, val_losses, timestamp, losses_dir


if __name__ == '__main__':

    # Load CASE 2 dataset
    X_train, y_train, X_val, y_val, X_test, y_test, norm_params, config = make_dataset(case_number=4)


    # Load pre-trained CASE 1 model
    benchmark_path = config["transfer_learning"]["benchmark_model"]
    base_model_type = config["model"]["type"]

    model = MODEL_ARCHITECTURES[base_model_type]()
    model.load_state_dict(torch.load(benchmark_path))

    # Freeze
    model = freeze_layers(model)

    # Train
    trained_model, model_path, train_losses, val_losses, timestamp, losses_dir = train_at_ht_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model=model,
        lr=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        max_epochs=config["training"]["max_epochs"],
        patience=config["training"]["patience"]
    )

    # Plot losses
    plot_losses(trained_model, model_path, train_losses, val_losses, timestamp, losses_dir, save=True)