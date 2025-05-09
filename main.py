from nilmtk.dataset_converters import convert_refit, convert_ukdale
from nilmtk import DataSet
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("Logger setup was successful")
    return logger


def get_timestamp():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return timestamp


def refit_csv_to_h5(logger):
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


def load_downsample_data(dataset, house_id, appliance_name, start_date, end_date, logger):
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


def combine_and_sync(mains, appliance, logger):
    logger.info("Combining mains and appliance data into a single DataFrame.")
    df = pd.DataFrame({
        'aggregate': mains.values.flatten(),
        'appliance': appliance.values.flatten()
    }, index=mains.index)
    logger.info("Data combined successfully.")
    return df


def normalize_data(df, logger):
    logger.info("Normalizing data.")

    aggregate_min = df['aggregate'].min()
    aggregate_max = df['aggregate'].max()
    df['aggregate'] = (df['aggregate'] - aggregate_min) / (aggregate_max - aggregate_min)
    appliance_max = df['appliance'].max()
    df['appliance'] = df['appliance'] / appliance_max

    logger.info("Normalization completed.")
    return df, aggregate_min, aggregate_max, appliance_max


def normalize_with_given_params(df, agg_min, agg_max, app_max, logger):
    logger.info("Normalizing data using training parameters.")
    df['aggregate'] = (df['aggregate'] - agg_min) / (agg_max - agg_min)
    df['appliance'] = df['appliance'] / app_max
    logger.info("Normalization using training parameters completed.")
    return df


def pad_aggregate_data(aggregate, window_size, logger):
    logger.info(f"Padding aggregate data with window size {window_size}.")
    half_window = window_size // 2
    pad_before = half_window
    pad_after = window_size - half_window - 1
    aggregate_padded = np.pad(aggregate, (pad_before, pad_after), 'constant')
    logger.info(f"Padding completed: {pad_before} before, {pad_after} after.")
    return aggregate_padded


def create_sliding_windows(aggregate_padded, appliance_values, window_size, window_step, logger):
    logger.info(f"Creating sliding windows with size {window_size} and step {window_step}.")
    X = []
    y = []
    for i in range(0, len(appliance_values), window_step):
        window = aggregate_padded[i:i+window_size]
        X.append(window)
        y.append(appliance_values[i])
    logger.info(f"Created {len(X)} sliding windows.")
    return np.array(X), np.array(y)


class STMModel(nn.Module):
    def __init__(self, input_channels=1, seq_len=100):
        super(STMModel, self).__init__()
        self.seq_len = seq_len

        # --- Spatial features with small kernel ---
        self.spatial_small = nn.Sequential(
            nn.Conv1d(input_channels, 20, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(20, 30, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(30, 30, kernel_size=5, stride=1, padding='same'),
            nn.ReLU()
        )

        # --- Spatial features with large kernel ---
        self.spatial_large = nn.Sequential(
            nn.Conv1d(input_channels, 20, kernel_size=10, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(20, 30, kernel_size=10, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(30, 30, kernel_size=10, stride=1, padding='same'),
            nn.ReLU()
        )

        # --- Temporal features with BiGRU ---
        self.bigru1 = nn.GRU(input_size=1, hidden_size=16, batch_first=True, bidirectional=True)
        self.bigru2 = nn.GRU(input_size=32, hidden_size=32, batch_first=True, bidirectional=True)
        self.bigru3 = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)

        # --- CBAM after concatenation ---
        self.cbam = CBAM(in_channels=30 + 30 + 128)  # 30 (small) + 30 (large) + 128 (BiGRU last layer)

        # --- Output Module (zgodny z Fig. 5) ---
        self.fc1 = nn.Linear((30 + 30 + 128) * seq_len, 1024)  # 188 * seq_len
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)
        self.output_linear = nn.Linear(1024, 1)
        self.output_sigmoid = nn.Linear(1024, 1)

    def forward(self, x):
        batch_size, _, seq_len = x.shape

        # Spatial features
        spatial_small = self.spatial_small(x)
        spatial_large = self.spatial_large(x)

        # Temporal features
        x_temp = x.permute(0, 2, 1)  # (B, T, C)
        out1, _ = self.bigru1(x_temp)
        out2, _ = self.bigru2(out1)
        out3, _ = self.bigru3(out2)
        temporal = out3.permute(0, 2, 1)  # (B, C, T)

        # Concatenate all features
        features = torch.cat([spatial_small, spatial_large, temporal], dim=1)

        # Attention mechanism
        Ft = self.cbam(features)  # (B, 188, T)

        # Output module (zgodnie z Fig. 5)
        flat = torch.flatten(Ft, start_dim=1)  # (B, 188 * T)
        fc = self.relu(self.fc1(flat))  # (B, 1024)

        linear_out = self.dropout(self.output_linear(fc))           # (B, 1)
        sigmoid_out = self.dropout(torch.sigmoid(self.output_sigmoid(fc)))  # (B, 1)

        output = linear_out * sigmoid_out  # element-wise multiplication

        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)  # shape: (B, C, 1)
        return x * attention  # Multiply by channel attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)  # shape: (B, 2, T)
        attention = self.sigmoid(self.conv(concat))  # shape: (B, 1, T)
        return x * attention  # Multiply by channel attention


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


MODEL_ARCHITECTURES = {
    'STMModel': STMModel
}


def train_stm_model(X_train, y_train, X_val, y_val, model, logger, lr=0.0001, batch_size=16, max_epochs=20, patience=2):
    logger.info("Preparing training and validation datasets...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Convert to tensors and move to device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)  # shape: (B, 1, T)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)  # shape: (B, 1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)  # shape: (B, 1, T)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)  # shape: (B, 1)

    logger.info(f"X_train_tensor: {X_train_tensor.shape}")
    logger.info(f"y_train_tensor: {y_train_tensor.shape}")
    logger.info(f"X_val_tensor: {X_val_tensor.shape}")
    logger.info(f"y_val_tensor: {y_val_tensor.shape}")

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    # File path to save model
    os.makedirs("models", exist_ok=True)
    timestamp = get_timestamp()
    model_path = f"models/{timestamp}_best_model.pth"

    logger.info("Starting training...")

    for epoch in range(max_epochs):
        logger.info(f"Epoch {epoch+1}/{max_epochs} - Training...")
        model.train()
        epoch_loss = 0.0

        for xb, yb in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch {epoch+1} - Training loss: {avg_train_loss:.6f}")

        # Validation
        logger.info(f"Epoch {epoch+1} - Validating...")
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False):
                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logger.info(f"Epoch {epoch+1} - Validation loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            logger.info(f"Epoch {epoch+1} - New best model found! Saving to {model_path}")
            best_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, model_path)
        else:
            epochs_no_improve += 1
            logger.info(f"Epoch {epoch+1} - No improvement. Patience: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break
    logger.info(f'Ended training.')
    return model, model_path, train_losses, val_losses, timestamp


if __name__ == '__main__':
    logger = setup_logger()
    dataset = refit_csv_to_h5(logger)

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
        end_date=end_date_train,
        logger=logger)
    df_train = combine_and_sync(mains_train, appliance_train, logger)
    df_train, agg_min, agg_max, app_max = normalize_data(df_train, logger)
    aggregate_padded_train = pad_aggregate_data(df_train['aggregate'].values, window_size, logger)
    X_train, y_train = create_sliding_windows(aggregate_padded_train, df_train['appliance'].values, window_size,
                                              window_step, logger)

    mains_test, appliance_test = load_downsample_data(
        dataset=dataset,
        house_id=house_id,
        appliance_name=appliance_name,
        start_date=start_date_test,
        end_date=end_date_test)
    df_test = combine_and_sync(mains_test, appliance_test, logger)
    df_test = normalize_with_given_params(df_test, agg_min, agg_max, app_max, logger)
    aggregate_padded_test = pad_aggregate_data(df_test['aggregate'].values, window_size, logger)
    X_test, y_test = create_sliding_windows(aggregate_padded_test, df_test['appliance'].values, window_size,
                                            window_step, logger)
    model = MODEL_ARCHITECTURES['STMModel']()
    trained_model, model_path, train_losses, val_losses, timestamp = train_stm_model(X_train, y_train, X_test, y_test, model, logger)
