from nilmtk.dataset_converters import convert_refit, convert_ukdale
from nilmtk import DataSet
from nilmtk.utils import dict_to_html, print_dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import seaborn as sns

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)


def refit_csv_to_h5():
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


def load_downsample_data(dataset, house_id, appliance_name, start_date, end_date):
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


def combine_and_sync(mains, appliance):
    logger.info("Combining mains and appliance data into a single DataFrame.")
    df = pd.DataFrame({
        'aggregate': mains.values.flatten(),
        'appliance': appliance.values.flatten()
    }, index=mains.index)
    logger.info("Data combined successfully.")
    return df


def normalize_data(df):
    logger.info("Normalizing data.")

    aggregate_min = df['aggregate'].min()
    aggregate_max = df['aggregate'].max()
    df['aggregate'] = (df['aggregate'] - aggregate_min) / (aggregate_max - aggregate_min)
    appliance_max = df['appliance'].max()
    df['appliance'] = df['appliance'] / appliance_max

    logger.info("Normalization completed.")
    return df, aggregate_min, aggregate_max, appliance_max


def normalize_with_given_params(df, agg_min, agg_max, app_max):
    logger.info("Normalizing data using training parameters.")
    df['aggregate'] = (df['aggregate'] - agg_min) / (agg_max - agg_min)
    df['appliance'] = df['appliance'] / app_max
    logger.info("Normalization using training parameters completed.")
    return df


def pad_aggregate_data(aggregate, window_size):
    logger.info(f"Padding aggregate data with window size {window_size}.")
    half_window = window_size // 2
    pad_before = half_window
    pad_after = window_size - half_window - 1
    aggregate_padded = np.pad(aggregate, (pad_before, pad_after), 'constant')
    logger.info(f"Padding completed: {pad_before} before, {pad_after} after.")
    return aggregate_padded, half_window


def create_sliding_windows(aggregate_padded, appliance_values, window_size, window_step, half_window):
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
    dataset = refit_csv_to_h5()
    house_id = 9
    appliance_name = 'fridge'
    window_size = 100
    window_step = 1

    start_date_train = '2015-01-01'
    end_date_train = '2015-07-01'
    start_date_test = '2014-12-01'
    end_date_test = '2014-12-10'

    mains_train, appliance_train = load_downsample_data(
        dataset=dataset,
        house_id=house_id,
        appliance_name=appliance_name,
        start_date=start_date_train,
        end_date=end_date_train)
    df_train = combine_and_sync(mains_train, appliance_train)
    df_train, agg_min, agg_max, app_max = normalize_data(df_train)
    aggregate_padded_train, half_window = pad_aggregate_data(df_train['aggregate'].values, window_size)
    X_train, y_train = create_sliding_windows(aggregate_padded_train, df_train['appliance'].values, window_size, window_step, half_window)

    mains_test, appliance_test = load_downsample_data(
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
    X_train, y_train, X_test, y_test, app_max = main()