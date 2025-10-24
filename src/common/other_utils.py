import logging
import os
import re
import yaml
import matplotlib.pyplot as plt
import torch
from datetime import datetime


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


