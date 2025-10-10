# from nilmtk.dataset_converters import convert_refit, convert_ukdale
# from nilmtk import DataSet
# from nilmtk.utils import dict_to_html, print_dict
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
from sklearn.model_selection import train_test_split

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.path.join(sys.path[0], project_dir))
from src.model.utils import inspect_h5_contents, setup_logger







def generate_seq2point_data(data_dict, sequence_length=599, target_appliance='fridge'):
    """
    Generates input (X) and output (y) sequences for a one-to-one seq2point NILM model.
    Assumes data_dict contains exactly one house.

    :param data_dict: Dictionary with filtered data for a single house
    :param sequence_length: Length of input sequence (default: 599)
    :param target_appliance: Target appliance name (default: 'fridge')
    :return: Tuple (X, y), where:
             - X is numpy array of shape (num_windows, sequence_length, 1)
             - y is numpy array of shape (num_windows,)
    """
    if len(data_dict) != 1:
        raise ValueError("Expected data_dict to contain exactly one house")

    house_id = next(iter(data_dict))

    if 'aggregate' not in data_dict[house_id]:
        raise ValueError(f"'aggregate' not found for house {house_id}")

    if target_appliance not in data_dict[house_id]:
        raise ValueError(f"'{target_appliance}' not found for house {house_id}")

    aggregate_series = data_dict[house_id]['aggregate'].dropna()
    target_series = data_dict[house_id][target_appliance].dropna()

    aligned = aggregate_series.join(target_series, how='inner', lsuffix='_agg', rsuffix='_target')
    aggregate_values = aligned.iloc[:, 0].values
    target_values = aligned.iloc[:, 1].values

    t0 = sequence_length // 2
    num_samples = len(aligned)

    X = []
    y = []

    for i in range(t0, num_samples - t0):
        window = aggregate_values[i - t0:i + t0 + 1]
        if len(window) == sequence_length:
            X.append(window.reshape(-1, 1))
            y.append(target_values[i])

    return np.array(X), np.array(y)





def filter_data_dict_by_time(data_dict, start_time, end_time):
    """
    Filters data_dict to include only data within the specified time range,
    resamples all time series to 8-second intervals, and drops missing values.
    Assumes data_dict contains exactly one house.

    :param data_dict: Dictionary returned by load_refit_csv_file, containing a single house
    :param start_time: Start time as a string, e.g., '2014-08-01'
    :param end_time: End time as a string, e.g., '2014-08-10'
    :return: Filtered and resampled data_dict with the same structure
    """
    from pandas import to_datetime

    if len(data_dict) != 1:
        raise ValueError("Expected data_dict to contain exactly one house")

    house_id = next(iter(data_dict))
    start = to_datetime(start_time)
    end = to_datetime(end_time)

    filtered = {house_id: {}}
    for appliance, df in data_dict[house_id].items():
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        # Filter by time range
        df = df[(df.index >= start) & (df.index <= end)]

        # df = df.resample("8S").mean().interpolate("linear").fillna(method='bfill').fillna(method='ffill')

        df = df.interpolate("linear").fillna(method="bfill").fillna(method="ffill")
        # Resample to 8-second intervals
        # df = df.resample("8S").mean()
        #
        # # Drop NaN values caused by resampling or missing data
        # df = df.dropna()

        filtered[house_id][appliance] = df

    return filtered


def generate_seq2point_data(data_dict, sequence_length=599, target_appliance='fridge'):
    """
    Generates input (X) and output (y) sequences for a one-to-one seq2point NILM model.
    Assumes data_dict contains exactly one house.

    :param data_dict: Dictionary with filtered data for a single house
    :param sequence_length: Length of input sequence (default: 599)
    :param target_appliance: Target appliance name (default: 'fridge')
    :return: Tuple (X, y), where:
             - X is numpy array of shape (num_windows, sequence_length, 1)
             - y is numpy array of shape (num_windows,)
    """
    if len(data_dict) != 1:
        raise ValueError("Expected data_dict to contain exactly one house")

    house_id = next(iter(data_dict))

    if 'aggregate' not in data_dict[house_id]:
        raise ValueError(f"'aggregate' not found for house {house_id}")

    if target_appliance not in data_dict[house_id]:
        raise ValueError(f"'{target_appliance}' not found for house {house_id}")

    aggregate_series = data_dict[house_id]['aggregate'].dropna()
    target_series = data_dict[house_id][target_appliance].dropna()

    aligned = aggregate_series.join(target_series, how='inner', lsuffix='_agg', rsuffix='_target')
    aggregate_values = aligned.iloc[:, 0].values
    target_values = aligned.iloc[:, 1].values

    t0 = sequence_length // 2
    num_samples = len(aligned)

    X = []
    y = []

    for i in range(t0, num_samples - t0):
        window = aggregate_values[i - t0:i + t0 + 1]
        if len(window) == sequence_length:
            X.append(window.reshape(-1, 1))
            y.append(target_values[i])

    return np.array(X), np.array(y)



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


def pad_data(data, window_size):
    logger = setup_logger()
    logger.info(f"Padding data with window size {window_size}.")
    half_window = window_size // 2
    pad_before = half_window
    pad_after = window_size - half_window - 1
    padded = np.pad(data, (pad_before, pad_after), 'constant')
    logger.info(f"Padding completed: {pad_before} before, {pad_after} after.")
    return padded


def normalize(X, y):
    """
    Min-max normalization for input (X) and output (y).
    Returns normalized data and the min/max values for later denormalization.
    """
    X_min = X.min()
    X_max = X.max()
    y_min = y.min()
    y_max = y.max()

    # Avoid division by zero
    X_range = X_max - X_min if X_max != X_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    X_norm = (X - X_min) / X_range
    y_norm = (y - y_min) / y_range

    norm_params = {
        "X_min": X_min,
        "X_max": X_max,
        "y_min": y_min,
        "y_max": y_max
    }

    return X_norm, y_norm, norm_params


def train_val_test_split(
        X, y,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        include_val=True,
        random_state=42
):
    """
    Splits data into train/val/test or train/test sets based on proportions.

    :param X: Input features (numpy array)
    :param y: Target values (numpy array)
    :param train_size: Proportion for training set (e.g., 0.6)
    :param val_size: Proportion for validation set (e.g., 0.2)
    :param test_size: Proportion for test set (e.g., 0.2)
    :param include_val: Whether to include a validation set
    :param random_state: Random seed for reproducibility
    :return:
        If include_val=True:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        If include_val=False:
            (X_train, X_test, y_train, y_test)
    """
    if include_val:
        total = train_size + val_size + test_size
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"train + val + test must sum to 1.0, but got {total}")

        # Split off training set
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1 - train_size), random_state=random_state
        )

        # Relative ratio of val vs test in remaining data
        val_ratio = val_size / (val_size + test_size)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    else:
        total = train_size + test_size
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"train + test must sum to 1.0, but got {total}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test


def create_sliding_windows(aggregate_padded, appliance_values, window_size, window_step):
    logger = setup_logger()
    logger.info(f"Creating sliding windows with size {window_size} and step {window_step}.")
    X = []
    y = []
    for i in range(0, len(appliance_values), window_step):
        window = aggregate_padded[i:i + window_size]
        X.append(window)
        y.append(appliance_values[i])
    logger.info(f"Created {len(X)} sliding windows.")
    return np.array(X), np.array(y)




def create_sliding_windows_with_normalization(aggregate_padded, appliance_values, window_size, window_step):
    """
    Creates sliding windows from aggregate power data with per-window normalization.
    Normalizes appliance values per-window by dividing by max value in the window.

    Parameters:
    -----------
    aggregate_padded : np.ndarray
        1D array of aggregate power readings (not padded anymore).
    appliance_values : np.ndarray
        1D array of appliance power readings (same length as aggregate_padded).
    window_size : int
        Length of each sliding window.
    window_step : int
        Step size for sliding the window.

    Returns:
    --------
    X : np.ndarray
        Normalized aggregate windows of shape (num_windows, window_size).
    y : np.ndarray
        Normalized middle appliance value per window (shape: (num_windows,)).
    agg_min_max : np.ndarray
        Per-window [min, max] of aggregate power (shape: (num_windows, 2)).
    app_max : np.ndarray
        Per-window max appliance value used for normalization (shape: (num_windows,)).
    """
    logger = setup_logger()
    logger.info(f"Creating sliding windows with size {window_size} and step {window_step}, with per-window normalization.")

    X = []
    y = []
    agg_min_max = []
    app_max = []

    half_window = window_size // 2

    for i in range(0, len(aggregate_padded) - window_size + 1, window_step):
        agg_window = aggregate_padded[i:i + window_size]
        app_window = appliance_values[i:i + window_size]

        # Normalize aggregate window
        agg_min = agg_window.min()
        agg_max = agg_window.max()
        if agg_max - agg_min != 0:
            agg_norm = (agg_window - agg_min) / (agg_max - agg_min)
        else:
            agg_norm = np.zeros_like(agg_window)

        # Normalize appliance window using max in this window
        app_max_val = app_window.max()
        if app_max_val == 0:
            app_max_val = 1e-6  # avoid division by zero

        app_window_norm = app_window / app_max_val

        # Save normalized data
        X.append(agg_norm)
        y.append(app_window_norm[half_window])  # middle point
        agg_min_max.append([agg_min, agg_max])
        app_max.append(app_max_val)

    logger.info(f"Created {len(X)} normalized sliding windows.")

    return (
        np.array(X),
        np.array(y),
        np.array(agg_min_max),
        np.array(app_max)
    )


def make_dataset1():
    data_dict = load_refit_csv_to_memory(
        csv_folder='datasets/refit',
        appliance_map_path='datasets/metadata/refit_appliance_map.json')
    

    # Train Dataset
    mains_res_train, appl_res_train = load_downsample_data_from_memory(
        data_dict=data_dict,
        house_id=9,
        appliance_name='fridge',
        start_date='2015-01-01',
        end_date='2015-07-01',
        resample_rate='30S')

    df_train_comb = combine_and_sync(
        mains=mains_res_train,
        appliance=appl_res_train)

    df_train_pad_agg = pad_data(
        data=df_train_comb['aggregate'].values,
        window_size=100)

    df_train_pad_app = pad_data(
        data=df_train_comb['appliance'].values,
        window_size=100)

    X_train_full, y_train_full, agg_min_max_train, app_max_train = create_sliding_windows_with_normalization(
        aggregate_padded=df_train_pad_agg,
        appliance_values=df_train_pad_app,
        window_size=100,
        window_step=1)

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train_full,
    #     y_train_full,
    #     test_size=0.1,
    #     random_state=42,
    #     shuffle=True
    # )

    # Test Dataset
    mains_res_test, appl_res_test = load_downsample_data_from_memory(
        data_dict=data_dict,
        house_id=9,
        appliance_name='fridge',
        start_date='2014-12-01',
        end_date='2014-12-10',
        resample_rate='30S')

    df_test_comb = combine_and_sync(
        mains=mains_res_test,
        appliance=appl_res_test)

    df_test_pad_agg = pad_data(
        data=df_test_comb['aggregate'].values, window_size=100)

    df_test_pad_app = pad_data(
        data=df_test_comb['appliance'].values, window_size=100)

    X_test, y_test, agg_min_max_test, app_max_test = create_sliding_windows_with_normalization(
        aggregate_padded=df_test_pad_agg,
        appliance_values=df_test_pad_app,
        window_size=100,
        window_step=1)

    return X_train_full, y_train_full, X_test, y_test, app_max_test


def make_dataset2():
    data_dict = load_refit_csv_file(
        csv_path='datasets/refit/CLEAN_House11.csv',
        appliance_map_path='datasets/metadata/refit_appliance_map.json')
    data_filtered = filter_data_dict_by_time(
        data_dict=data_dict,
        start_time='2014-07-30',
        end_time='2014-08-14')
    X, y = generate_seq2point_data(
        data_dict=data_filtered,
        sequence_length=599,
        target_appliance='fridge_freezer'
    )
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        include_val=True
    )
    X_train_norm, y_train_norm, norm_params = normalize(X_train, y_train)

    X_val_norm = (X_val - norm_params["X_min"]) / (norm_params["X_max"] - norm_params["X_min"])
    y_val_norm = (y_val - norm_params["y_min"]) / (norm_params["y_max"] - norm_params["y_min"])

    X_test_norm = (X_test - norm_params["X_min"]) / (norm_params["X_max"] - norm_params["X_min"])
    y_test_norm = (y_test - norm_params["y_min"]) / (norm_params["y_max"] - norm_params["y_min"])

    return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, norm_params, X, y


if __name__ == '__main__':
    X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, norm_params, X, y = make_dataset2()
    # print("Train:", X_train_norm.shape, y_train_norm.shape)
    # print("Val:", X_val_norm.shape, y_val_norm.shape)
    # print("Test:", X_test_norm.shape, y_test_norm.shape)
    plt.hist(y_train_norm, bins=50)
    plt.title("Histogram y_train_norm")
    plt.show()

    plt.hist(y_val_norm, bins=50)
    plt.title("Histogram y_val_norm")
    plt.show()

    plt.hist(y_test_norm, bins=50)
    plt.title("Histogram y_val_norm")
    plt.show()

    plt.hist(y, bins=100)
    plt.title("Histogram y_train (before normalization)")
    plt.show()

    czas = np.arange(X.shape[0])
    plt.figure(figsize=(12, 6))
    for i in range(min(5, X.shape[1])):  # pokażemy do 5 pierwszych cech
        plt.plot(czas, X[:, i], label=f'Cech {i}')
    plt.title("Przebiegi cech X w czasie")
    plt.xlabel("Czas (próbki)")
    plt.ylabel("Wartości cech")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Przebieg y (docelowa zmienna)
    plt.figure(figsize=(12, 4))
    plt.plot(czas, y, label="y", color='orange')
    plt.title("Przebieg wartości y w czasie")
    plt.xlabel("Czas (próbki)")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()

    print(norm_params)







