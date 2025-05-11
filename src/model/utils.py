import matplotlib.pyplot as plt
import logging
import torch
import os
from datetime import datetime
import pandas as pd


def setup_logger():
    # Sprawdzamy, czy logger jest już ustawiony
    if not logging.getLogger().hasHandlers():
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Tworzymy handler i ustawiamy format
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)

        # Dodajemy handler do loggera
        logger.addHandler(handler)
        logger.info("Logger setup was successful.")
    else:
        logger = logging.getLogger()

    return logger


def get_timestamp():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return timestamp


def plot_losses(model, model_path, train_losses, val_losses, logger, timestamp, save=True):
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


def inspect_h5_contents(h5_path):
    logger = setup_logger()
    with pd.HDFStore(h5_path, mode='r') as store:
        logger.info(f"Contents of '{h5_path}':")
        for key in store.keys():
            logger.info(f"  {key}")


if __name__ == '__main__':
    print('XD')
