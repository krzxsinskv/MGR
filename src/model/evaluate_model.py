from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import torch
import logging
import sys
import os
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.path.join(sys.path[0], project_dir))
from src.model.models_architectures import MODEL_ARCHITECTURES
from src.common.other_utils import setup_logger, plot_predictions, extract_timestamp, plot_histogram
from src.common.data_utils import make_dataset


def evaluate_model(model, X_test, y_test, app_max, batch_size=16, timestamp=None):
    logger = setup_logger()
    logger.info("Starting evaluation...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Evaluating", leave=False):
            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    logger.info("Predicted values (y_pred):")
    logger.info(f"Shape: {y_pred.shape}, Min: {y_pred.min()}, Max: {y_pred.max()}, Mean: {y_pred.mean()}")

    logger.info("True values (y_true):")
    logger.info(f"Shape: {y_true.shape}, Min: {y_true.min()}, Max: {y_true.max()}, Mean: {y_true.mean()}")

    # Cofnięcie normalizacji
    y_pred = y_pred * app_max
    y_pred = np.clip(y_pred, 0, None)
    y_true = y_true * app_max

    # Metryki
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    sae = np.abs(np.sum(y_pred) - np.sum(y_true)) / np.sum(y_true)

    logger.info("Evaluation Metrics:")
    logger.info(f"MAE:  {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"SAE:  {sae:.4f}")

    print("Pred min/max", y_pred.min(), y_pred.max())
    print("True min/max", y_true.min(), y_true.max())

    plot_predictions(y_true, y_pred, 2000, timestamp, save=True)
    plot_histogram(y_pred, bins=50, timestamp=timestamp, save=True)

    return mae, rmse, sae


if __name__ == '__main__':
    _, _, _, _, X_test, y_test, norm_params = make_dataset()
    model = MODEL_ARCHITECTURES['STMModel']()
    model_path = 'models/2025-10-25_15-59_best_model.pth'
    model.load_state_dict(torch.load(model_path))
    timestamp = extract_timestamp(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Testowy kawałek danych
    X_tensor = torch.tensor(X_test[:32], dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        batch_size, _, seq_len = X_tensor.shape

        # Spatial features
        spatial_small = model.spatial_small(X_tensor)
        spatial_large = model.spatial_large(X_tensor)

        # Temporal features
        x_temp = X_tensor.permute(0, 2, 1)
        out1, _ = model.bigru1(x_temp)
        out2, _ = model.bigru2(out1)
        out3, _ = model.bigru3(out2)
        temporal = out3.permute(0, 2, 1)

        # Połączenie
        features = torch.cat([spatial_small, spatial_large, temporal], dim=1)
        Ft = model.cbam(features)

        # Flatten + fully connected
        flat = torch.flatten(Ft, start_dim=1)
        fc = model.relu(model.fc1(flat))

        linear_out = model.output_linear(fc)
        sigmoid_out = torch.sigmoid(model.output_sigmoid(fc))
        final_out = linear_out * sigmoid_out

        print("📊 Diagnostyka STMModel:")
        print("linear_out:  min/max/mean:",
              linear_out.min().item(), linear_out.max().item(), linear_out.mean().item())
        print("sigmoid_out: min/max/mean:",
              sigmoid_out.min().item(), sigmoid_out.max().item(), sigmoid_out.mean().item())
        print("final_out:   min/max/mean:",
              final_out.min().item(), final_out.max().item(), final_out.mean().item())

    evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        app_max=norm_params['appliance_max'],
        batch_size=16,
        timestamp=timestamp)
