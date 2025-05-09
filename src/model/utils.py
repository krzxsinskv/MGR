import matplotlib.pyplot as plt
import logging
import torch
import os
from datetime import datetime


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("Logger setup was successful")
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


def test(timestamp):
    save_dir = os.path.join("results", "losses")
    os.makedirs(save_dir, exist_ok=True)
    losses_path = f"results/losses/{timestamp}_losses.png"
    return save_dir, losses_path


if __name__ == '__main__':
    print('XD')
