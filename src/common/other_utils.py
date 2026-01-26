import logging
import os
import re
import yaml
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import numpy as np


def setup_logger():
    """
    Sets up the root logger to output messages to the console with a specific format.

    :param None
    :return: Configured root logger instance (logging.Logger)
    """
    logger = logging.getLogger()

    if getattr(logger, "_initialized", False):
        return logger

    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Logger setup was successful.")

    logger._initialized = True
    return logger


def get_timestamp():
    """
    Generates a timestamp string of the current date and time in the format YYYY-MM-DD_HH-MM.

    :param None
    :return: Timestamp string (str) in format "YYYY-MM-DD_HH-MM"
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return timestamp


def extract_timestamp(model_path):
    """
    Extracts a timestamp from a filename, matching the pattern YYYY-MM-DD_HH-MM.

    :param model_path: Path to the model file (str)
    :return: Extracted timestamp string (str) if found, otherwise None
    """
    filename = os.path.basename(model_path)
    match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}', filename)
    if match:
        return match.group(0)
    else:
        return None


def get_csv_paths_from_config(csv_folder, house_ids):
    """
    Zwraca listę ścieżek CSV z folderu odpowiadających podanym numerom domów.

    :param csv_folder: Ścieżka do folderu z plikami REFIT CSV
    :param house_ids: Lista numerów domów do załadowania (np. [9, 11]) lub pojedyncza liczba
    :return: Lista pasujących ścieżek plików CSV
    """
    logger = setup_logger()

    if isinstance(house_ids, int):
        house_ids = [house_ids]

    paths = []
    for hid in house_ids:
        expected_file = os.path.join(csv_folder, f"CLEAN_House{hid}.csv")
        if os.path.isfile(expected_file):
            paths.append(expected_file)
        else:
            logger.warning(f"⚠️ File for house {hid} not found at {expected_file}")
    return paths


def load_yaml_config(yaml_path):
    """
    Loads a YAML file from the given path and returns its contents as a dictionary.

    :param yaml_path: Path to the YAML file
    :return: dict containing the parsed YAML configuration
    :raises FileNotFoundError: if the YAML file does not exist
    :raises yaml.YAMLError: if there is an error parsing the YAML file
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)  # Use safe_load for security
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    return config


def merge_dicts(base, override):
    for k, v in override.items():
        if (
            k in base
            and isinstance(base[k], dict)
            and isinstance(v, dict)
        ):
            merge_dicts(base[k], v)
        else:
            base[k] = v
    return base


def load_case_config(case_number: int):
    base = load_yaml_config("configs/base.yaml")
    case_path = f"configs/training/case{case_number}.yaml"

    case_cfg = load_yaml_config(case_path)

    final_cfg = merge_dicts(base, case_cfg)
    return final_cfg


def load_eval_config(ds=None, app=None):
    ds_path = f"configs/datasets/{ds}.yaml"
    app_path = f"configs/appliances/{app}.yaml"

    base = load_yaml_config("configs/base.yaml")
    ds_cfg = load_yaml_config(ds_path)
    app_cfg = load_yaml_config(app_path)

    final_cfg = merge_dicts(base, ds_cfg)
    final_cfg = merge_dicts(final_cfg, app_cfg)
    return final_cfg


def plot_losses(model, model_path, train_losses, val_losses, timestamp, losses_dir, save=True):
    logger = setup_logger()
    logger.info('Plotting train and validation losses')

    # Load best model
    model.load_state_dict(torch.load(model_path))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid(True)

    if save:
        os.makedirs(losses_dir, exist_ok=True)
        save_path = os.path.join(losses_dir, f"{timestamp}_losses.png")
        plt.savefig(save_path)
        logger.info(f"Saved loss plot to {save_path}")

    plt.close()


def plot_predictions(y_true, y_pred, samples, timestamp, save=True):
    logger = setup_logger()
    logger.info('Plotting true and predicted values')
    plt.plot(y_true[:samples], label='True')
    plt.plot(y_pred[:samples], label='Predicted')
    plt.legend()
    plt.title("True and predicted values")
    if save:
        save_dir = os.path.join("results", "predictions")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{timestamp}_predictions.png")

        plt.savefig(save_path)
        logger.info(f"Saved predicted values plot to {save_path}")
        plt.show()
    else:
        plt.show()


def plot_histogram(y_pred, bins, timestamp, save=True):
    logger = setup_logger()
    logger.info('Plotting histogram of predicted values')
    plt.hist(y_pred, bins)
    plt.title("Histogram of Predicted Appliance Power")
    plt.xlabel("Power (W)")
    plt.ylabel("Frequency")
    plt.title("Histogram of predicted values")
    if save:
        save_dir = os.path.join("results", "histograms")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{timestamp}_histogram.png")

        plt.savefig(save_path)
        logger.info(f"Saved predicted values plot to {save_path}")
    else:
        plt.show()


def create_input_sample():
    T = 100
    x = np.zeros(T)
    np.random.seed(0)
    baseline_noise = np.random.normal(0.003, 0.0008, size=T)
    window = 5
    baseline_smoothed = np.convolve(baseline_noise, np.ones(window) / window, mode='same')
    x += baseline_smoothed
    x[25:35] += 0.39 + np.random.uniform(-0.01, 0.01, size=10)
    x[50] = 0.38

    return x


def extract_fusion_features(model, input_tensor):
    """
    Performs a forward pass through the STM model and returns the fusion
    feature map captured by the forward-pre-hook

    Parameters:
        model: STMModel instance with registered hooks
        input_tensor: torch.Tensor of shape (B, 1, seq_len)

    Returns:
        torch.Tensor: fusion feature map of shape (1, 188, T)
    """

    # Set model to evaluation mode (disables dropout, ensures deterministic behavior)
    model.eval()

    # Reset previously captured values
    model.captured["fusion_features"] = None

    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)

    return model.captured["fusion_features"]


def plot_fusion_feature_map(fusion_features, timestamp, save=True):
    """
    Plots a colored fusion feature map based on the extracted
    fusion features captured by the model hook.

    Parameters:
        fusion_features (torch.Tensor or np.ndarray):
            Tensor of shape (1, 188, T) or squeezed to (188, T).
        timestamp (str):
            A string used to name the saved plot (e.g., datetime).
        save (bool):
            Whether to save the generated plot to disk.

    Returns:
        None
    """

    # Initialize logger
    logger = setup_logger()
    logger.info("Starting fusion feature map visualization...")

    # Convert tensor to numpy array
    if isinstance(fusion_features, torch.Tensor):
        logger.info("Converting tensor input to NumPy array.")
        fusion_features = fusion_features.squeeze().cpu().numpy()

    # Validate shape
    assert fusion_features.ndim == 2, "Fusion_features must be 2D after squeeze()."
    logger.info(f"Fusion feature map shape: {fusion_features.shape}")

    # Begin plotting
    logger.info("Generating heatmap...")
    plt.figure(figsize=(8, 6), dpi=120)

    plt.imshow(
        fusion_features,
        aspect='auto',
        cmap="inferno",           # closest to article style
        interpolation='nearest'
    )

    plt.colorbar(label="Feature intensity")
    plt.title("Fusion Feature Map (Fig. 9 Style)", fontsize=14)
    plt.xlabel("Time steps", fontsize=12)
    plt.ylabel("Feature channels", fontsize=12)
    plt.grid(False)
    plt.tight_layout()

    # Save or show
    if save:
        save_dir = os.path.join("results", "fusion_feature_maps")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{timestamp}_ffmap.png")

        plt.savefig(save_path)
        logger.info(f"Saved fusion feature map to: {save_path}")
    else:
        logger.info("Displaying fusion feature map without saving.")
        plt.show()


def save_metrics_to_txt(
    mae,
    rmse,
    sae,
    y_pred,
    y_true,
    timestamp,
    energy_true_kwh=None,     # NEW
    energy_pred_kwh=None,     # NEW
    save_dir="results/metrics"
):
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{timestamp}_metrics.txt"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "w") as f:
        f.write("Evaluation Metrics\n")
        f.write("===================\n\n")

        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"SAE:  {sae:.4f}\n\n")

        f.write("Prediction Ranges\n")
        f.write("=================\n")
        f.write(f"Predicted min: {y_pred.min():.6f}\n")
        f.write(f"Predicted max: {y_pred.max():.6f}\n")
        f.write(f"True min:      {y_true.min():.6f}\n")
        f.write(f"True max:      {y_true.max():.6f}\n")

        if energy_true_kwh is not None and energy_pred_kwh is not None:
            f.write("Energy Consumption\n")
            f.write("===================\n")
            f.write(f"True energy (kWh):      {energy_true_kwh:.6f}\n")
            f.write(f"Predicted energy (kWh): {energy_pred_kwh:.6f}\n")

    return filepath


def plot_channel_attention_map(channel_map, timestamp, save=True):
    """
    Plots the channel attention map (Fig. 10a style).

    Parameters:
        channel_map (torch.Tensor or np.ndarray):
            Shape (C, 1) or squeeze() -> (C,).
        timestamp (str):
            Identifier for saved filename.
        save (bool):
            Save or just display.

    Returns:
        None
    """
    logger = setup_logger()
    logger.info("Starting channel attention map visualization...")

    # Convert to numpy
    if isinstance(channel_map, torch.Tensor):
        logger.info("Converting tensor input to NumPy array.")
        channel_map = channel_map.squeeze().cpu().numpy()

    # Ensure shape is (C, 1) or (C,)
    assert channel_map.ndim in (1, 2), "Channel map must be 1D or 2D after squeeze()."
    if channel_map.ndim == 1:
        channel_map = channel_map[:, None]

    logger.info(f"Channel attention map shape: {channel_map.shape}")

    # Plot
    plt.figure(figsize=(4, 8), dpi=120)
    plt.imshow(
        channel_map,
        aspect='auto',
        cmap="inferno",
        interpolation='nearest'
    )
    plt.colorbar(label="Attention weight")
    plt.title("Channel Attention Map (Fig. 10a)", fontsize=14)
    plt.xlabel("Attention")
    plt.ylabel("Channels")
    plt.tight_layout()

    # Save
    if save:
        save_dir = os.path.join("results", "channel_attention_maps")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{timestamp}_channel_attention.png")

        plt.savefig(save_path)
        logger.info(f"Saved channel attention map to: {save_path}")
    else:
        logger.info("Displaying channel attention map without saving.")
        plt.show()


def plot_spatial_attention_map(spatial_map, timestamp, save=True):
    """
    Plots the spatial attention map (Fig. 10c style).

    Parameters:
        spatial_map (torch.Tensor or np.ndarray):
            Shape (1, T) or squeeze() -> (T,).
        timestamp (str)
        save (bool)

    Returns:
        None
    """
    logger = setup_logger()
    logger.info("Starting spatial attention map visualization...")

    if isinstance(spatial_map, torch.Tensor):
        logger.info("Converting tensor input to NumPy array.")
        spatial_map = spatial_map.squeeze().cpu().numpy()

    assert spatial_map.ndim in (1, 2), "Spatial map must be 1D or 2D after squeeze()."
    if spatial_map.ndim == 1:
        spatial_map = spatial_map[None, :]

    logger.info(f"Spatial attention map shape: {spatial_map.shape}")

    plt.figure(figsize=(10, 3), dpi=120)
    plt.imshow(
        spatial_map,
        aspect='auto',
        cmap="inferno",
        interpolation='nearest'
    )
    plt.colorbar(label="Attention weight")
    plt.title("Spatial Attention Map (Fig. 10c)", fontsize=14)
    plt.xlabel("Time steps")
    plt.ylabel("Spatial attention")
    plt.tight_layout()

    if save:
        save_dir = os.path.join("results", "spatial_attention_maps")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{timestamp}_spatial_attention.png")

        plt.savefig(save_path)
        logger.info(f"Saved spatial attention map to: {save_path}")
    else:
        logger.info("Displaying spatial attention map without saving.")
        plt.show()


def plot_post_attention_feature_map(feature_map, timestamp, save=True, stage="post_channel"):
    """
    Plots fusion features after channel or spatial attention
    (Fig. 10b or Fig. 10d style).

    Parameters:
        feature_map (torch.Tensor or np.ndarray):
            Shape (C, T)
        timestamp (str)
        save (bool)
        stage (str): "post_channel" or "post_spatial"

    Returns:
        None
    """
    logger = setup_logger()
    logger.info(f"Starting {stage} feature map visualization...")

    if isinstance(feature_map, torch.Tensor):
        logger.info("Converting tensor input to NumPy array.")
        feature_map = feature_map.squeeze().cpu().numpy()

    assert feature_map.ndim == 2, "Feature map must be 2D (C, T)."
    logger.info(f"{stage} map shape: {feature_map.shape}")

    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(
        feature_map,
        aspect='auto',
        cmap="inferno",
        interpolation='nearest'
    )
    plt.colorbar(label="Feature intensity")
    plt.title(
        f"Fusion Features After {stage.replace('_', ' ').title()} (Fig. 10)",
        fontsize=14
    )
    plt.xlabel("Time steps")
    plt.ylabel("Feature channels")
    plt.tight_layout()

    if save:
        save_dir = os.path.join("results", f"{stage}_maps")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{timestamp}_{stage}.png")

        plt.savefig(save_path)
        logger.info(f"Saved {stage} feature map to: {save_path}")
    else:
        logger.info(f"Displaying {stage} feature map without saving.")
        plt.show()
