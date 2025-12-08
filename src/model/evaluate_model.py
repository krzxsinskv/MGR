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
from src.common.other_utils import setup_logger, plot_predictions, extract_timestamp, plot_histogram, save_metrics_to_txt
from src.common.data_utils import make_dataset


def evaluate_model(
    model,
    X_test,
    y_test,
    params,
    method="minmax",
    batch_size=16,
    timestamp=None,
):
    logger = setup_logger()
    logger.info(f"Starting evaluation using normalization method: {method}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=batch_size
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Evaluating", leave=False):
            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    logger.info(f"Predicted (normalized) range: {y_pred.min():.4f} – {y_pred.max():.4f}")
    logger.info(f"True (normalized) range:     {y_true.min():.4f} – {y_true.max():.4f}")

    app_max = params["appliance_max"]

    if method == "minmax":
        y_pred = y_pred * app_max
        y_true = y_true * app_max

    elif method == "clipped_quantile_minmax":
        y_pred = y_pred * app_max
        y_true = y_true * app_max

    elif method == "clipped_value_minmax":
        y_pred = y_pred * app_max
        y_true = y_true * app_max

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Optional clipping after denormalization
    # y_pred = np.clip(y_pred, 0, None)

    # Metrics
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    sae = np.abs(np.sum(y_pred) - np.sum(y_true)) / np.sum(y_true)

    logger.info("Evaluation Metrics:")
    logger.info(f"MAE:  {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"SAE:  {sae:.4f}")

    print("Pred min/max", y_pred.min(), y_pred.max())
    print("True min/max", y_true.min(), y_true.max())

    metrics_path = save_metrics_to_txt(
        mae=mae,
        rmse=rmse,
        sae=sae,
        y_pred=y_pred,
        y_true=y_true,
        timestamp=timestamp
    )

    logger.info(f"Saved metrics to {metrics_path}")

    plot_predictions(y_true, y_pred, 2000, timestamp, save=True)
    plot_histogram(y_pred, bins=50, timestamp=timestamp, save=True)

    return mae, rmse, sae


if __name__ == '__main__':
    _, _, _, _, X_test, y_test, norm_params, config = make_dataset(case_number=1)
    model = MODEL_ARCHITECTURES[config["model"]["type"]]()
    model_path = config["evaluation"]["model"]
    model.load_state_dict(torch.load(model_path))
    timestamp = extract_timestamp(model_path)

    evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        params=norm_params,
        method=config["normalization"]["method"],
        batch_size=config["training"]["batch_size"],
        timestamp=timestamp)
