import matplotlib.pyplot as plt
import logging
import torch
import os
from datetime import datetime
import pandas as pd
import re


def setup_logger():
    """
    Sets up the root logger to output messages to the console with a specific format.

    Behavior:
    - Ensures logger is configured only once (even if called multiple times).
    - Clears any existing handlers on first setup (useful in environments like Google Colab).
    - Adds a StreamHandler with a custom formatter.
    - Sets the logging level to INFO.
    - Logs "Logger setup was successful." only on the first call.

    Returns:
        logging.Logger: Configured root logger instance
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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return timestamp


def extract_timestamp(model_path):
    filename = os.path.basename(model_path)
    match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}', filename)
    if match:
        return match.group(0)
    else:
        return None


def plot_losses(model, model_path, train_losses, val_losses, timestamp, save=True):
    logger = setup_logger()
    logger.info('Plotting train and validation losses')
    model.load_state_dict(torch.load(model_path))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Loss over Epochs")
    if save:
        save_dir = os.path.join("results", "losses")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{timestamp}_losses.png")

        plt.savefig(save_path)
        logger.info(f"Saved loss plot to {save_path}")
        plt.show()
    else:
        plt.show()


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
        plt.show()
    else:
        plt.show()


def inspect_h5_contents(h5_path):
    logger = setup_logger()
    with pd.HDFStore(h5_path, mode='r') as store:
        logger.info(f"Contents of '{h5_path}':")
        for key in store.keys():
            logger.info(f"  {key}")


if __name__ == '__main__':
    model_path = 'models/2025-05-27_16-21_best_model.pth'
    timestamp = extract_timestamp(model_path)
    print(timestamp)
