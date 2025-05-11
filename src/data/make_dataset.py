from nilmtk.dataset_converters import convert_refit, convert_ukdale
from nilmtk import DataSet
from nilmtk.utils import dict_to_html, print_dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional

import os
import zipfile
import requests
import logging
import os
import zipfile
import requests
import time
import numpy as np
import webbrowser
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import json


project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.path.join(sys.path[0], project_dir))
from src.model.utils import inspect_h5_contents, setup_logger

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
# ch.setFormatter(formatter)
# logger.addHandler(ch)


def refit_csv_to_h5_nilmtk():
    logger = setup_logger()
    dataset_path = "datasets/h5/refit.h5"

    if os.path.exists(dataset_path):
        logger.info(f"File {dataset_path} found.")
    else:
        logger.info(f"File {dataset_path} not found. Starting data conversion.")

        input_path = "datasets/refit"
        if not os.path.exists(input_path):
            logger.error(f"Invalid input path: {input_path}. Cannot perform conversion.")
            return

        convert_refit(input_path, dataset_path)

    data = DataSet(dataset_path)
    logger.info(f"Dataset loaded from {dataset_path}.")
    return data


def refit_csv_to_h5(csv_folder, output_h5_path, appliance_map_path):
    logger = setup_logger()
    if os.path.exists(output_h5_path):
        logger.info(f"File '{output_h5_path}' already exists. Skipping conversion.")
        return

    logger.info(f"Loading appliance mapping from '{appliance_map_path}'")
    with open(appliance_map_path) as f:
        appliance_mapping = json.load(f)

    logger.info(f"Starting REFIT CSV to HDF5 conversion from folder: {csv_folder}")

    with pd.HDFStore(output_h5_path, mode='w') as store:
        for file in os.listdir(csv_folder):
            if file.endswith(".csv"):
                house_name = os.path.splitext(file)[0]  # e.g., CLEAN_House1
                house_id = house_name.split("House")[-1]  # extract number
                file_path = os.path.join(csv_folder, file)

                logger.info(f"Reading house data: {file_path} (house {house_id})")

                try:
                    df = pd.read_csv(file_path, parse_dates=[0], index_col=0)
                    logger.info(f"Loaded {house_name}, shape: {df.shape}")

                    for col in df.columns:
                        key = col.strip().lower()
                        device_name = appliance_mapping.get(house_id, {}).get(key, key)
                        dataset_path = f"house{house_id}/{device_name}"
                        store.put(dataset_path, df[[col]])
                        logger.info(f"Saved {dataset_path}")
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")

    logger.info(f"REFIT dataset successfully saved to '{output_h5_path}'")


def load_downsample_data_nilmtk(dataset, house_id, appliance_name, start_date, end_date):
    logger = setup_logger()
    logger.info(f"Setting data window from {start_date} to {end_date}.")
    dataset.set_window(start=start_date, end=end_date)

    logger.info(f"Loading data for house {house_id}, appliance '{appliance_name}'.")
    building = dataset.buildings[house_id]
    elec = building.elec
    mains = elec.mains()
    appliance = elec[appliance_name]

    logger.info(f"Concatenating mains readings.")
    mains = pd.concat(mains.load(), axis=0)
    mains.index = pd.to_datetime(mains.index)
    mains_resample = mains.resample('30S').mean().interpolate(method='time')

    logger.info(f"Concatenating {appliance_name} readings.")
    appliance = pd.concat(appliance.load(), axis=0)
    appliance.index = pd.to_datetime(appliance.index)
    appliance_resample = appliance.resample('30S').mean().interpolate(method='time')

    logger.info("Data loading and resampling completed.")
    return mains_resample, appliance_resample


def load_downsample_data(h5_path, house_id, appliance_name, start_date, end_date, resample_rate='30S'):
    logger = setup_logger()
    logger.info(f"Loading data for house {house_id}, appliance '{appliance_name}', window: {start_date} to {end_date}")

    with pd.HDFStore(h5_path, mode='r') as store:
        try:
            mains_path = f'house{house_id}/aggregate'
            appliance_path = f'house{house_id}/{appliance_name.lower()}'

            logger.info(f"Reading mains from {mains_path}")
            mains = store.get(mains_path)
            logger.info(f"Reading appliance from {appliance_path}")
            appliance = store.get(appliance_path)
        except KeyError as e:
            logger.error(f"Data not found: {e}")
            return None, None

    # Ensure datetime index
    mains.index = pd.to_datetime(mains.index)
    appliance.index = pd.to_datetime(appliance.index)

    # Cut to time window
    mains = mains[start_date:end_date]
    appliance = appliance[start_date:end_date]

    logger.info(f"Downsampling mains and appliance data to {resample_rate}")
    mains_resample = mains.resample(resample_rate).mean().interpolate(method='time')
    appliance_resample = appliance.resample(resample_rate).mean().interpolate(method='time')

    logger.info("Data loading and resampling completed.")
    return mains_resample, appliance_resample


def combine_and_sync(mains, appliance):
    logger = setup_logger()
    logger.info("Combining mains and appliance data into a single DataFrame.")
    df = pd.DataFrame({
        'aggregate': mains.values.flatten(),
        'appliance': appliance.values.flatten()
    }, index=mains.index)
    logger.info("Data combined successfully.")
    return df


def normalize_data(df):
    logger = setup_logger()
    logger.info("Normalizing data.")

    aggregate_min = df['aggregate'].min()
    aggregate_max = df['aggregate'].max()
    df['aggregate'] = (df['aggregate'] - aggregate_min) / (aggregate_max - aggregate_min)
    appliance_max = df['appliance'].max()
    df['appliance'] = df['appliance'] / appliance_max

    logger.info("Normalization completed.")
    return df, aggregate_min, aggregate_max, appliance_max


def normalize_with_given_params(df, agg_min, agg_max, app_max):
    logger = setup_logger()
    logger.info("Normalizing data using training parameters.")
    df['aggregate'] = (df['aggregate'] - agg_min) / (agg_max - agg_min)
    df['appliance'] = df['appliance'] / app_max
    logger.info("Normalization using training parameters completed.")
    return df


def pad_aggregate_data(aggregate, window_size):
    logger = setup_logger()
    logger.info(f"Padding aggregate data with window size {window_size}.")
    half_window = window_size // 2
    pad_before = half_window
    pad_after = window_size - half_window - 1
    aggregate_padded = np.pad(aggregate, (pad_before, pad_after), 'constant')
    logger.info(f"Padding completed: {pad_before} before, {pad_after} after.")
    return aggregate_padded


def create_sliding_windows(aggregate_padded, appliance_values, window_size, window_step):
    logger = setup_logger()
    logger.info(f"Creating sliding windows with size {window_size} and step {window_step}.")
    X = []
    y = []
    for i in range(0, len(appliance_values), window_step):
        window = aggregate_padded[i:i+window_size]
        X.append(window)
        y.append(appliance_values[i])
    logger.info(f"Created {len(X)} sliding windows.")
    return np.array(X), np.array(y)


# def create_sliding_windows(aggregate_padded, appliance_values, window_size, window_step, half_window):
#     logger.info(f"Creating sliding windows with size {window_size} and step {window_step}.")
#     X = []
#     y = []
#
#     max_index = len(aggregate_padded) - window_size + 1
#     num_windows = min(len(appliance_values), max_index)
#
#     for i in range(0, num_windows, window_step):
#         window = aggregate_padded[i:i + window_size]
#         if len(window) == window_size:
#             X.append(window)
#             y.append(appliance_values[i])
#         else:
#             logger.warning(f"Skipped incomplete window at index {i}")
#
#     logger.info(f"Created {len(X)} sliding windows.")
#     return np.array(X), np.array(y)


def main():
    dataset = refit_csv_to_h5_nilmtk()
    house_id = 9
    appliance_name = 'fridge'
    window_size = 100
    window_step = 1

    start_date_train = '2015-01-01'
    end_date_train = '2015-07-01'
    start_date_test = '2014-12-01'
    end_date_test = '2014-12-10'

    mains_train, appliance_train = load_downsample_data_nilmtk(
        dataset=dataset,
        house_id=house_id,
        appliance_name=appliance_name,
        start_date=start_date_train,
        end_date=end_date_train)
    df_train = combine_and_sync(mains_train, appliance_train)
    df_train, agg_min, agg_max, app_max = normalize_data(df_train)
    aggregate_padded_train, half_window = pad_aggregate_data(df_train['aggregate'].values, window_size)
    X_train, y_train = create_sliding_windows(aggregate_padded_train, df_train['appliance'].values, window_size, window_step, half_window)

    mains_test, appliance_test = load_downsample_data_nilmtk(
        dataset=dataset,
        house_id=house_id,
        appliance_name=appliance_name,
        start_date=start_date_test,
        end_date=end_date_test)
    df_test = combine_and_sync(mains_test, appliance_test)
    df_test = normalize_with_given_params(df_test, agg_min, agg_max, app_max)
    aggregate_padded_test, half_window = pad_aggregate_data(df_test['aggregate'].values, window_size)
    X_test, y_test = create_sliding_windows(aggregate_padded_test, df_test['appliance'].values, window_size, window_step, half_window)
    return X_train, y_train, X_test, y_test, app_max


if __name__ == '__main__':
    # X_train, y_train, X_test, y_test, app_max = main()
    # logger = setup_logger()
    refit_csv_to_h5('datasets/refit', 'datasets/h5/refit_new.h5', 'datasets/metadata/refit_metadata.json')
    inspect_h5_contents('datasets/h5/refit_new.h5')
    # mains_resample, appliance_resample = load_downsample_data('datasets/h5/refit_new.h5', 9, 'fridge', '2015-01-01', '2015-07-01', resample_rate='30S')
    # df_train = combine_and_sync(mains_resample, appliance_resample)
    # df_train, agg_min, agg_max, app_max = normalize_data(df_train)
    # aggregate_padded_train = pad_aggregate_data(df_train['aggregate'].values, window_size=100)
    # X_train, y_train = create_sliding_windows(aggregate_padded_train, df_train['appliance'].values, window_size=100, window_step=1)
