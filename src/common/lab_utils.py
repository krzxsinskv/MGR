import h5py
import numpy as np
from pathlib import Path
import os
import sys

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.path.join(sys.path[0], project_dir))

from src.common.other_utils import setup_logger


def load_lab_mat(path, appliance_row=10):
    """
    Loads MATLAB v7.3 (.mat, HDF5) laboratory file.
    appliance_row: zero-based index (kettle = 10)
    """
    logger = setup_logger()
    logger.info(f"Loading file: {path}")

    with h5py.File(path, "r") as f:
        opis = f["opis"]

        aggregate = np.array(opis["moc"]).squeeze()
        P = np.array(opis["features"]["P"])
        appliance = P[:, appliance_row]

    aggregate = aggregate.astype(np.float32)
    appliance = appliance.astype(np.float32)

    neg_agg = np.sum(aggregate < 0)
    neg_app = np.sum(appliance < 0)

    aggregate[aggregate < 0] = 0.0
    appliance[appliance < 0] = 0.0

    logger.info(
        f"Loaded aggregate shape={aggregate.shape}, "
        f"appliance shape={appliance.shape}, "
        f"negative values clipped: aggregate={neg_agg}, appliance={neg_app}"
    )

    return aggregate, appliance


def split_train_val_xy(X, y, val_ratio=0.1):
    n_total = len(X)
    split_idx = int(n_total * (1 - val_ratio))
    logger = setup_logger()

    logger.info(
        f"Splitting dataset with val_ratio={val_ratio:.2f}: "
        f"total={n_total}, train={split_idx}, val={n_total - split_idx}"
    )

    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(
        f"Train shapes: X_train={X_train.shape}, y_train={y_train.shape} | "
        f"Val shapes: X_val={X_val.shape}, y_val={y_val.shape}"
    )

    return X_train, X_val, y_train, y_val


def create_windowed_samples(aggregate, appliance, window_length=100, ratio=10):
    logger = setup_logger()
    logger.info(
        f"Creating windows: "
        f"window_length={window_length}, ratio={ratio}"
    )

    X, y = [], []
    half_blocks = window_length // ratio // 2

    skipped = 0

    for k in range(0, len(aggregate) - window_length + 1, ratio):
        i = k // ratio + (half_blocks - 1)

        if i < 0 or i >= len(appliance):
            skipped += 1
            continue

        X.append(aggregate[k:k + window_length])
        y.append(appliance[i])

    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32)

    logger.info(
        f"Generated {len(X)} samples "
        f"(skipped {skipped} due to boundary conditions)"
    )

    return X, y


def build_lab_dataset(mat_paths, appliance_row=10):
    logger = setup_logger()
    logger.info(
        f"Building dataset from {len(mat_paths)} files, "
        f"appliance_row={appliance_row}"
    )

    X_all, y_all = [], []

    for path in mat_paths:
        agg, app = load_lab_mat(path, appliance_row=appliance_row)
        X, y = create_windowed_samples(agg, app, 100, 10)

        logger.info(
            f"File {path}: X shape={X.shape}, y shape={y.shape}"
        )

        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    logger.info(
        f"Final dataset shape: X={X_all.shape}, y={y_all.shape}"
    )

    return X_all, y_all


def compute_norm_params_xy(X_train, y_train):
    logger = setup_logger()
    params = {
        "aggregate_min": X_train.min(),
        "aggregate_max": X_train.max(),
        "appliance_max": y_train.max(),
    }

    logger.info(
        "Normalization parameters computed: "
        f"aggregate_min={params['aggregate_min']:.3f}, "
        f"aggregate_max={params['aggregate_max']:.3f}, "
        f"appliance_max={params['appliance_max']:.3f}"
    )

    return params


def apply_norm_xy(X, y, params):
    logger = setup_logger()
    Xn = (X - params["aggregate_min"]) / (
        params["aggregate_max"] - params["aggregate_min"]
    )
    yn = y / params["appliance_max"]

    logger.info(
        f"Applied normalization: "
        f"X range=({Xn.min():.3f}, {Xn.max():.3f}), "
        f"y range=({yn.min():.3f}, {yn.max():.3f})"
    )

    return Xn.astype(np.float32), yn.astype(np.float32)


def make_lab_dataset():
    X_train_val, y_train_val = build_lab_dataset(
        ['datasets/lab/train1.mat', 'datasets/lab/train2.mat', 'datasets/lab/train3.mat', 'datasets/lab/train4.mat'],
        10)
    X_test, y_test = build_lab_dataset(['datasets/lab/test.mat'], 10)
    X_train, X_val, y_train, y_val = split_train_val_xy(X_train_val, y_train_val, 0.1)
    params = compute_norm_params_xy(X_train, y_train)
    X_train_norm, y_train_norm = apply_norm_xy(X_train, y_train, params)
    X_val_norm, y_val_norm = apply_norm_xy(X_val, y_val, params)
    X_test_norm, y_test_norm = apply_norm_xy(X_test, y_test, params)

    return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, params


if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test, params = make_lab_dataset()
