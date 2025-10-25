import matplotlib.pyplot as plt
import logging
import torch
import os
from datetime import datetime
import pandas as pd
import re

















if __name__ == '__main__':
    model_path = 'models/2025-05-27_16-21_best_model.pth'
    timestamp = extract_timestamp(model_path)
    print(timestamp)
