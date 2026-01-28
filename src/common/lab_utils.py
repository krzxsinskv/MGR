from scipy.io import loadmat
import numpy as np
from pathlib import Path
import os
import sys

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.path.join(sys.path[0], project_dir))


def load_lab_mat(path, appliance_row=10):
    """
    Loads laboratory .mat file and extracts:
    - aggregate power (opis.moc)
    - appliance power (opis.features.P[appliance_row])

    appliance_row: zero-based index (kettle = 10)
    """
    mat = loadmat(path, squeeze_me=True)

    opis = mat["opis"]

    aggregate = np.asarray(opis["moc"]).astype(np.float32)
    appliance = np.asarray(opis["features"]["P"])[appliance_row].astype(np.float32)

    return aggregate, appliance


if __name__ == '__main__':
    load_lab_mat