import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.path.join(sys.path[0], project_dir))
from src.common.other_utils import setup_logger
from src.common.other_utils import get_csv_paths_from_config, load_case_config


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
    Always includes the 'Aggregate' and 'Issues' columns if present.

    :param appliance_map_path: Path to a JSON file mapping appliance names
    :param csv_paths: List of CSV file paths to load
    :param appliances: (optional) List of appliances to load; if None, loads all appliances
    :return: Dictionary with structure:
             {
                 house_id (str): {
                     device_name (str): pandas.DataFrame
                     'aggregate': pandas.DataFrame
                     'issues': pandas.DataFrame
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

                # Always include 'aggregate' and 'issues' if present
                if appliances and device_name not in appliances and device_name.lower() not in ["aggregate", "issues"]:
                    continue

                data_dict[house_id][device_name] = df[[col]]
                logger.info(f"Loaded data for house{house_id}/{device_name}")

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")

    logger.info("REFIT CSV loading complete.")
    return data_dict


def prepare_data(
    data_dict,
    house_id,
    appliance_name,
    start_date,
    end_date,
    resample_rate="30S",
    clear_issues=True
):
    """
    Prepares in-memory REFIT data for a given house and appliance.
    Ensures time alignment, trims data to the specified date range,
    resamples to a uniform frequency, interpolates missing values,
    and optionally clears 'Issues' samples.

    Parameters
    ----------
    data_dict : dict
        Dictionary returned by `load_refit_csv`, containing data organized by house and appliance.
    house_id : int or str
        ID of the house to process.
    appliance_name : str
        Name of the appliance (lowercase recommended).
    start_date : str
        Start date for slicing (e.g. '2015-01-01').
    end_date : str
        End date for slicing (e.g. '2015-07-01').
    resample_rate : str, optional
        Pandas-compatible resampling rate string (default '30S').
    clear_issues : bool, optional
        If True, clears samples where Issues == 1 and appliance > mains.
        Should be True for training/validation, False for testing.

    Returns
    -------
    mains_resampled, appliance_resampled : pd.DataFrame
        Resampled and aligned dataframes indexed by datetime.
    """
    logger = setup_logger()
    logger.info(f"Preparing data for house {house_id}, appliance '{appliance_name}' (clear_issues={clear_issues})")

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

        # Optionally clear issues (only for train/val)
        # if clear_issues:
        #     issues = house_data.get("issues")
        #     if issues is not None:
        #         issues = issues.loc[start_date:end_date]
        #         mask_issue = issues['Issues'] == 1
        #         mask_bad = mask_issue & (appliance.iloc[:, 0] > mains.iloc[:, 0])
        #
        #         n_cleared = mask_bad.sum()
        #         total_samples = len(mask_bad)
        #         percent_cleared = (n_cleared / total_samples * 100) if total_samples > 0 else 0.0
        #
        #         # Appliance == 0 in faulty samples
        #         appliance = appliance.copy()
        #         appliance.loc[mask_bad, appliance.columns[0]] = 0
        #
        #         logger.info(
        #             f"Cleared {n_cleared} issue samples "
        #             f"({percent_cleared:.2f}% of total, Issues==1 & appliance>mains)."
        #         )
        # ALL ISSUES == 1
        if clear_issues:
            issues = house_data.get("issues")
            if issues is not None:
                issues = issues.loc[start_date:end_date]
                mask_issue = issues["Issues"] == 1

                n_removed = mask_issue.sum()
                total = len(mask_issue)
                percent = (n_removed / total * 100) if total > 0 else 0.0

                logger.info(
                    f"Removing {n_removed} samples ({percent:.2f}%) due to Issues==1 "
                    "(REFIT: unreliable submetering readings)."
                )

                # Remove these samples from both mains and appliance
                mains = mains.loc[~mask_issue]
                appliance = appliance.loc[~mask_issue]

        if mains.empty or appliance.empty:
            raise ValueError(f"No data in time window {start_date}–{end_date} for house {house_id}")

        # Resample and interpolate
        logger.info(f"Resampling data to {resample_rate}")
        mains_resampled = mains.resample(resample_rate).mean().interpolate(method='time').dropna()
        appliance_resampled = appliance.resample(resample_rate).mean().interpolate(method='time').dropna()

        logger.info(
            f"Data preparation completed successfully: {len(mains_resampled)} samples "
            f"({start_date} → {end_date})"
        )
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


def split_train_val(df_train_val, val_ratio=0.1):
    """
    Splits a combined mains+appliance DataFrame into train and validation sets, preserving time order.

    Parameters
    ----------
    df_train_val : pd.DataFrame
        Combined DataFrame containing 'aggregate' and 'appliance' columns (from combine_and_sync).
    val_ratio : float, optional
        Proportion of data to allocate to the validation set (default 0.1 = 10%).

    Returns
    -------
    df_train : pd.DataFrame
        Training subset of df_train_val.
    df_val : pd.DataFrame
        Validation subset of df_train_val.
    """
    logger = setup_logger()
    if not isinstance(df_train_val, pd.DataFrame):
        raise ValueError("Input df_train_val must be a pandas DataFrame.")

    if 'aggregate' not in df_train_val.columns or 'appliance' not in df_train_val.columns:
        raise ValueError("DataFrame must contain 'aggregate' and 'appliance' columns.")

    total_rows = len(df_train_val)
    if total_rows == 0:
        raise ValueError("Input DataFrame is empty.")

    split_idx = int(total_rows * (1 - val_ratio))
    df_train = df_train_val.iloc[:split_idx]
    df_val = df_train_val.iloc[split_idx:]

    logger.info(f"Splitting dataset: {len(df_train)} train samples, {len(df_val)} validation samples.")
    logger.info(f"Validation ratio: {val_ratio:.2f} (time-based split).")
    logger.info(f"Train range: {df_train.index[0]} → {df_train.index[-1]}")
    logger.info(f"Val   range: {df_val.index[0]} → {df_val.index[-1]}")

    return df_train, df_val


import numpy as np
import pandas as pd


def compute_normalization_params(df, method="minmax", clip_quantile=0.995, clip_value=5000):
    """
    Compute parameters required for different normalization strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'aggregate' and 'appliance' columns.
    method : str
        Normalization method. Supported:
            - "minmax"
            - "clipped_quantile_minmax"
            - "clipped_value_minmax"
    clip_quantile : float, optional
        Used only when method='clipped_quantile_minmax'.
        Defines quantile threshold for clipping.
    clip_value : float, optional
        Used only when method='clipped_value_minmax'.
        Explicit clipping value.

    Returns
    -------
    params : dict
        Dictionary containing normalization parameters.
    """
    logger = setup_logger()
    logger.info(f"Computing normalization params using method: {method}")

    if 'aggregate' not in df.columns or 'appliance' not in df.columns:
        raise ValueError("DataFrame must contain 'aggregate' and 'appliance' columns.")

    params = {}

    # --- MIN-MAX normalization ---
    if method == "minmax":
        params["aggregate_min"] = df["aggregate"].min()
        params["aggregate_max"] = df["aggregate"].max()

    # --- CLIPPED QUANTILE MIN-MAX ---
    elif method == "clipped_quantile_minmax":
        clip_val = df["aggregate"].quantile(clip_quantile)
        params["aggregate_min"] = 0.0
        params["aggregate_max"] = clip_val
        params["clip_quantile"] = clip_quantile
        logger.info(f"Clipping aggregate at {clip_quantile*100:.2f}% → {clip_val:.3f}")

    # --- CLIPPED VALUE MIN-MAX ---
    elif method == "clipped_value_minmax":
        if clip_value is None:
            raise ValueError("clip_value must be provided when using 'clipped_value_minmax'.")

        params["aggregate_min"] = 0.0
        params["aggregate_max"] = float(clip_value)
        params["clip_value"] = float(clip_value)
        logger.info(f"Clipping aggregate using explicit value → {clip_value:.3f}")

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Appliance normalization (literature standard)
    params["appliance_max"] = df["appliance"].max()
    logger.info(f"Params computed: {params}")

    return params


def apply_normalization(df, params, method="minmax"):
    """
    Apply normalization based on precomputed parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'aggregate' and 'appliance' columns.
    params : dict
        Normalization parameters returned by compute_normalization_params().
    method : str
        Normalization method. Supported:
            - "minmax"
            - "clipped_quantile_minmax"
            - "clipped_value_minmax"

    Returns
    -------
    df_norm : pd.DataFrame
        DataFrame with added columns:
            - 'aggregate_norm'
            - 'appliance_norm'
    """
    logger = setup_logger()
    logger.info(f"Applying normalization using method: {method}")

    if 'aggregate' not in df.columns or 'appliance' not in df.columns:
        raise ValueError("DataFrame must contain 'aggregate' and 'appliance' columns.")

    df_norm = df.copy()

    # --- AGGREGATE NORMALIZATION ---
    if method == "minmax":
        df_norm["aggregate_norm"] = (
            (df["aggregate"] - params["aggregate_min"]) /
            (params["aggregate_max"] - params["aggregate_min"])
        )

    elif method == "clipped_quantile_minmax":
        clip_val = params["aggregate_max"]
        clipped = np.clip(df["aggregate"], 0, clip_val)
        df_norm["aggregate_norm"] = clipped / clip_val

    elif method == "clipped_value_minmax":
        clip_val = params["aggregate_max"]  # clip_value stored here
        clipped = np.clip(df["aggregate"], 0, clip_val)
        df_norm["aggregate_norm"] = clipped / clip_val

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # --- APPLIANCE NORMALIZATION ---
    df_norm["appliance_norm"] = df["appliance"] / params["appliance_max"]

    return df_norm


def create_windowed_samples(df, window_length=100):
    """
    Generates windowed samples and labels from aggregate and appliance power data
    according to the sliding window and zero-padding method described in the paper (Fig. 2).

    For each sample:
        - Input:  aggregate power values in a window of length `window_length`
        - Label:  appliance power value at the midpoint of that window

    Padding:
        - If window_length is even: pad with (w/2) zeros before and (w/2 - 1) zeros after
        - If window_length is odd:  pad with (w-1)/2 zeros before and after
    Sliding step is set to 1.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'aggregate' and 'appliance' columns (after normalization).
    window_length : int, optional
        Length of each sliding window (default = 100).

    Returns
    -------
    X : np.ndarray
        2D array of shape (num_samples, window_length) containing aggregate power windows.
    y : np.ndarray
        1D array of shape (num_samples,) containing appliance power labels.
    """
    logger = setup_logger()
    logger.info("Starting sample window generation...")
    logger.info(f"Input DataFrame shape: {df.shape}")

    if 'aggregate' not in df.columns or 'appliance' not in df.columns:
        if 'aggregate_norm' in df.columns and 'appliance_norm' in df.columns:
            logger.info("Detected normalized columns ('aggregate_norm', 'appliance_norm'). Renaming for processing.")
            df = df.rename(columns={'aggregate_norm': 'aggregate', 'appliance_norm': 'appliance'})
        else:
            raise ValueError(
                "DataFrame must contain 'aggregate'/'appliance' or 'aggregate_norm'/'appliance_norm' columns."
            )

    aggregate = df['aggregate'].values
    appliance = df['appliance'].values
    w = window_length

    if w % 2 == 0:
        pad_before = w // 2
        pad_after = w // 2 - 1
        logger.info(f"Even window length detected ({w}). Padding: before={pad_before}, after={pad_after}")
    else:
        pad_before = pad_after = (w - 1) // 2
        logger.info(f"Odd window length detected ({w}). Symmetric padding: {pad_before} before and after")

    aggregate_padded = np.pad(aggregate, (pad_before, pad_after), 'constant', constant_values=0)
    logger.info(f"Aggregate padded length: {len(aggregate_padded)} (original: {len(aggregate)})")

    num_samples = len(appliance)
    X = np.zeros((num_samples, w), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.float32)

    for k in range(num_samples):
        X[k, :] = aggregate_padded[k: k + w]
        y[k] = appliance[k]

    logger.info(f"Generated {num_samples} samples with window length = {w}.")
    logger.info(f"Final shapes: X = {X.shape}, y = {y.shape}")
    logger.info("Window generation completed successfully.")

    return X, y


def make_dataset(case_number: int):
    config = load_case_config(case_number)

    csv_paths = get_csv_paths_from_config(
        csv_folder=os.path.join("datasets", config["data"]["dataset"]),
        house_ids=config["data"]["houses"])

    data = load_refit_csv(
        appliance_map_path=config["paths"]["appliance_map"],
        csv_paths=csv_paths,
        appliances=config["data"]["appliances"])

    mains_train_val, appliance_train_val = prepare_data(
        data_dict=data,
        house_id=config["data"]["houses"][0],
        appliance_name=config["data"]["appliances"][0],
        start_date=config["data"]["train_range"][0],
        end_date=config["data"]["train_range"][1],
        resample_rate=config["data"]["resample_rate"],
        clear_issues=config["data"]["clear_issues"]
    )
    mains_test, appliance_test = prepare_data(
        data_dict=data,
        house_id=config["data"]["houses"][0],
        appliance_name=config["data"]["appliances"][0],
        start_date=config["data"]["test_range"][0],
        end_date=config["data"]["test_range"][1],
        resample_rate=config["data"]["resample_rate"],
        clear_issues=config["data"]["clear_issues"]
    )

    df_train_val = combine_and_sync(mains=mains_train_val, appliance=appliance_train_val)
    df_test = combine_and_sync(mains=mains_test, appliance=appliance_test)

    df_train, df_val = split_train_val(df_train_val=df_train_val, val_ratio=config["data"]["val_ratio"])

    norm_params = compute_normalization_params(df=df_train, method=config["normalization"]["method"])

    df_train_norm = apply_normalization(df=df_train, params=norm_params, method=config["normalization"]["method"])
    df_val_norm = apply_normalization(df=df_val, params=norm_params, method=config["normalization"]["method"])
    df_test_norm = apply_normalization(df=df_test, params=norm_params, method=config["normalization"]["method"])

    X_train, y_train = create_windowed_samples(
        df_train_norm[['aggregate_norm', 'appliance_norm']],
        window_length=config["data"]["window_size"]
    )
    X_val, y_val = create_windowed_samples(
        df_val_norm[['aggregate_norm', 'appliance_norm']],
        window_length=config["data"]["window_size"]
    )
    X_test, y_test = create_windowed_samples(
        df_test_norm[['aggregate_norm', 'appliance_norm']],
        window_length=config["data"]["window_size"]
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, norm_params, config


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test, norm_params_train, config = make_dataset(case_number=1)
    print(norm_params_train)
