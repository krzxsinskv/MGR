import json
import os
import sys
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.path.join(sys.path[0], project_dir))
from src.common.other_utils import setup_logger
from src.common.other_utils import get_csv_paths_from_config, load_yaml_config


def load_refit_csv_to_memory(csv_folder, appliance_map_path):
    """
    Loads multiple REFIT CSV files from a folder into memory and organizes data by house and appliance.
    Applies an appliance mapping defined in a JSON file.

    :param csv_folder: Path to the folder containing REFIT CSV files
    :param appliance_map_path: Path to a JSON file mapping appliance names
                               (per house) to standardized names
    :return: Dictionary with structure:
             {
                 house_id (str): {
                     device_name (str): pandas.DataFrame
                 }
             }
             Each DataFrame contains time-indexed readings for a single appliance.
             Includes data for all houses found in the given folder.
    """
    logger = setup_logger()

    logger.info(f"Loading appliance mapping from '{appliance_map_path}'")
    with open(appliance_map_path) as f:
        appliance_mapping = json.load(f)

    logger.info(f"Starting REFIT CSV in-memory loading from folder: {csv_folder}")

    data_dict = {}

    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            house_name = os.path.splitext(file)[0]  # e.g., CLEAN_House1
            house_id = house_name.split("House")[-1]  # extract number
            file_path = os.path.join(csv_folder, file)

            logger.info(f"Reading house data: {file_path} (house {house_id})")

            try:
                df = pd.read_csv(file_path, parse_dates=[0], index_col=0)
                logger.info(f"Loaded {house_name}, shape: {df.shape}")

                if house_id not in data_dict:
                    data_dict[house_id] = {}

                for col in df.columns:
                    key = col.strip().lower()
                    device_name = appliance_mapping.get(house_id, {}).get(key, key)
                    data_dict[house_id][device_name] = df[[col]]
                    logger.info(f"Loaded data for house{house_id}/{device_name}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

    logger.info("REFIT dataset successfully loaded into memory.")
    return data_dict


def load_refit_csv_file(csv_path, appliance_map_path):
    """
    Loads a single REFIT CSV file into memory and organizes data by appliance.
    Applies an appliance mapping defined in a JSON file.

    :param csv_path: Path to a single REFIT CSV file
    :param appliance_map_path: Path to a JSON file mapping appliance names
                               (per house) to standardized names
    :return: Dictionary with structure:
             {
                 house_id (str): {
                     device_name (str): pandas.DataFrame
                 }
             }
             Each DataFrame contains time-indexed readings for a single appliance.
             Returns an empty dictionary if the file cannot be processed.
    """
    logger = setup_logger()

    logger.info(f"Loading appliance mapping from '{appliance_map_path}'")
    with open(appliance_map_path) as f:
        appliance_mapping = json.load(f)

    if not os.path.isfile(csv_path):
        logger.error(f"CSV path does not exist or is not a file: {csv_path}")
        return {}

    house_name = os.path.splitext(os.path.basename(csv_path))[0]  # e.g., CLEAN_House11
    house_id = house_name.split("House")[-1]

    logger.info(f"Reading data from: {csv_path} (house {house_id})")

    data_dict = {house_id: {}}

    try:
        df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
        logger.info(f"Loaded {house_name}, shape: {df.shape}")

        for col in df.columns:
            key = col.strip().lower()
            device_name = appliance_mapping.get(house_id, {}).get(key, key)
            data_dict[house_id][device_name] = df[[col]]
            logger.info(f"Loaded data for house{house_id}/{device_name}")
    except Exception as e:
        logger.error(f"Failed to process {csv_path}: {e}")

    logger.info("REFIT CSV successfully loaded into memory.")
    return data_dict


def load_refit_csv(appliance_map_path, csv_paths, appliances=None):
    """
    Loads selected REFIT CSV files into memory and organizes data by house and appliance.
    Always includes the "Aggregate" energy consumption column if present.

    :param appliance_map_path: Path to a JSON file mapping appliance names
    :param csv_paths: List of CSV file paths to load
    :param appliances: (optional) List of appliances to load; if None, loads all appliances
    :return: Dictionary with structure:
             {
                 house_id (str): {
                     device_name (str): pandas.DataFrame
                 }
             }
    """
    logger = setup_logger()

    logger.info(f"Loading appliance mapping from '{appliance_map_path}'")
    with open(appliance_map_path) as f:
        appliance_mapping = json.load(f)

    data_dict = {}

    for file_path in csv_paths:
        house_name = os.path.splitext(os.path.basename(file_path))[0]
        house_id = house_name.split("House")[-1]

        try:
            df = pd.read_csv(file_path, parse_dates=[0], index_col=0)
            logger.info(f"Loaded file {file_path} for house {house_id} with shape {df.shape}")

            if house_id not in data_dict:
                data_dict[house_id] = {}

            for col in df.columns:
                key = col.strip().lower()
                device_name = appliance_mapping.get(house_id, {}).get(key, key)

                # Always include "Aggregate" if present
                if appliances and device_name not in appliances and device_name.lower() != "aggregate":
                    continue

                data_dict[house_id][device_name] = df[[col]]
                logger.info(f"Loaded data for house{house_id}/{device_name}")

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")

    logger.info("REFIT CSV loading complete.")
    return data_dict


if __name__ == '__main__':
    config = load_yaml_config(yaml_path="configs/article1_case1.yaml")

    csv_paths = get_csv_paths_from_config(
        csv_folder=os.path.join("datasets", config["data"]["dataset"]),
        house_ids=config["data"]["houses"])

    data = load_refit_csv(
        appliance_map_path=config["paths"]["appliance_map"],
        csv_paths=csv_paths,
        appliances=config["data"]["appliances"])

    # data_dict = load_refit_csv_file(
    #     csv_path='datasets/refit/CLEAN_House11.csv',
    #     appliance_map_path='datasets/metadata/refit_appliance_map.json')

    print(data)
