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


def prepare_data(data_dict, house_id, appliance_name, start_date, end_date, resample_rate="30S"):
    """
    Prepares in-memory REFIT data for a given house and appliance.
    Ensures time alignment, trims data to the specified date range,
    resamples to a uniform frequency, interpolates missing values,
    and optionally removes any remaining NaN samples.

    :param data_dict: Dictionary returned by `load_refit_csv`, containing data organized by house and appliance
    :param house_id: ID of the house to process (int or str)
    :param appliance_name: Name of the appliance (string, lowercase recommended)
    :param start_date: Start date for slicing (e.g. '2015-01-01')
    :param end_date: End date for slicing (e.g. '2015-07-01')
    :param resample_rate: Pandas-compatible resampling rate string (default '30S')
    :return: Tuple (mains_resampled, appliance_resampled), where each is a pandas.DataFrame
             indexed by datetime and resampled to the specified frequency
    """
    logger = setup_logger()
    logger.info(f"Preparing data for house {house_id}, appliance '{appliance_name}'")

    try:
        house_data = data_dict.get(str(house_id))
        if house_data is None:
            raise KeyError(f"House {house_id} not found in data_dict")

        mains = house_data.get("aggregate")
        appliance = house_data.get(appliance_name.lower())

        if mains is None or appliance is None:
            raise KeyError(f"Missing 'aggregate' or '{appliance_name}' for house {house_id}")

        # Ensure datetime index
        mains.index = pd.to_datetime(mains.index)
        appliance.index = pd.to_datetime(appliance.index)

        # Cut to time window
        mains = mains.loc[start_date:end_date]
        appliance = appliance.loc[start_date:end_date]

        if mains.empty or appliance.empty:
            raise ValueError(f"No data in time window {start_date}–{end_date} for house {house_id}")

        # Resample and interpolate
        logger.info(f"Resampling data to {resample_rate}")
        mains_resampled = mains.resample(resample_rate).mean().interpolate(method='time').dropna()
        appliance_resampled = appliance.resample(resample_rate).mean().interpolate(method='time').dropna()

        logger.info("Data preparation completed successfully.")
        return mains_resampled, appliance_resampled

    except Exception as e:
        logger.error(f"Failed to prepare data for house {house_id}, appliance '{appliance_name}': {e}")
        return None, None


def combine_and_sync(mains, appliance):
    """
    Combines mains and appliance power consumption data into a single, synchronized DataFrame.
    The resulting DataFrame shares a common datetime index and contains two columns:
    'aggregate' (for mains) and 'appliance' (for the selected device).

    :param mains: pandas.DataFrame or pandas.Series representing aggregate (mains) power data
    :param appliance: pandas.DataFrame or pandas.Series representing appliance power data
    :return: pandas.DataFrame with structure:
             index (DatetimeIndex)
             ├── aggregate (float)
             └── appliance (float)
    """
    logger = setup_logger()
    logger.info("Combining mains and appliance data into a single DataFrame.")

    df = pd.DataFrame({
        'aggregate': mains.values.flatten(),
        'appliance': appliance.values.flatten()
    }, index=mains.index)

    logger.info("Data combined successfully.")
    return df


if __name__ == '__main__':
    config = load_yaml_config(yaml_path="configs/article1_case1.yaml")

    csv_paths = get_csv_paths_from_config(
        csv_folder=os.path.join("datasets", config["data"]["dataset"]),
        house_ids=config["data"]["houses"])

    data = load_refit_csv(
        appliance_map_path=config["paths"]["appliance_map"],
        csv_paths=csv_paths,
        appliances=config["data"]["appliances"])

    # Train dataset

    mains_train, appliance_train = prepare_data(
        data_dict=data,
        house_id=config["data"]["houses"][0],
        appliance_name=config["data"]["appliances"][0],
        start_date=config["data"]["train_range"][0],
        end_date=config["data"]["train_range"][1],
        resample_rate=config["data"]["resample_rate"]
    )

    df = combine_and_sync(mains=mains_train, appliance=appliance_train)

    # data_dict = load_refit_csv_file(
    #     csv_path='datasets/refit/CLEAN_House11.csv',
    #     appliance_map_path='datasets/metadata/refit_appliance_map.json')

    print(df)
