import os
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.path.join(sys.path[0], project_dir))
from src.data.make_dataset import *


def inspect_mains_and_appliance(mains, appliance):
    """
    Inspect basic properties of mains and appliance time series.

    Parameters:
    -----------
    mains : pd.Series or pd.DataFrame
        The downsampled aggregate power data.

    appliance : pd.Series or pd.DataFrame
        The downsampled power data of a specific appliance.

    This function prints:
        - Length of each time series
        - Count of NaN values
        - First 5, middle 5, and last 5 data points
    """

    for data, label in [(mains, "Mains"), (appliance, "Appliance")]:
        print(f"\n===== Inspection Report: {label} =====")
        print(f"Length: {len(data)}")
        nan_count = data.isna().sum().values[0] if isinstance(data, pd.DataFrame) else data.isna().sum()
        print(f"NaN count: {nan_count}")

        if len(data) >= 5:
            print("\nFirst 5 values:")
            print(data.iloc[:5])

            mid_start = len(data) // 2 - 2
            mid_end = mid_start + 5
            print("\nMiddle 5 values:")
            print(data.iloc[mid_start:mid_end])

            print("\nLast 5 values:")
            print(data.iloc[-5:])
        else:
            print("\nData too short to show first/middle/last 5 values.")
            print(data)


def inspect_combined_df(df):
    """
    Inspect the combined DataFrame by printing the first, middle, and last 5 rows.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing both 'aggregate' and 'appliance' columns.

    This function prints:
        - Shape of the DataFrame
        - First 5 rows
        - Middle 5 rows
        - Last 5 rows
    """
    print("\n===== Inspection Report: Combined DataFrame =====")
    print(f"Shape: {df.shape}")

    if len(df) >= 5:
        print("\nFirst 5 rows:")
        print(df.head(5))

        mid_start = len(df) // 2 - 2
        mid_end = mid_start + 5
        print("\nMiddle 5 rows:")
        print(df.iloc[mid_start:mid_end])

        print("\nLast 5 rows:")
        print(df.tail(5))
    else:
        print("\nDataFrame too short to show first/middle/last 5 rows.")
        print(df)


def inspect_sliding_windows(X, y):
    """
    Inspect sliding window samples by printing the first, middle, and last windows
    along with their corresponding appliance values.

    Parameters:
    -----------
    X : np.ndarray
        Sliding windows of aggregate power (2D array of shape [n_windows, window_size]).
    y : np.ndarray
        Corresponding appliance power values (1D array of shape [n_windows]).
    """
    print("\n===== Sliding Windows Inspection =====")
    print(f"Total number of windows: {len(X)}")
    print(f"Window shape: {X[0].shape if len(X) > 0 else 'N/A'}")

    if len(X) >= 1:
        print("\n--- First Window ---")
        print(X[0])
        print("Appliance value:", y[0])

    if len(X) >= 3:
        mid_idx = len(X) // 2
        print("\n--- Middle Window ---")
        print(X[mid_idx])
        print("Appliance value:", y[mid_idx])

    if len(X) >= 1:
        print("\n--- Last Window ---")
        print(X[-1])
        print("Appliance value:", y[-1])


def inspect_sliding_window_with_normalization(X, y, window_min_max, max_appliance_value):
    """
    Inspect the content of sliding window output arrays: X, y, window_min_max, and max_appliance_value.

    Parameters:
    -----------
    X : np.ndarray
        Array of shape (num_windows, window_size) with normalized aggregate windows.
    y : np.ndarray
        Array of shape (num_windows,) with normalized appliance values.
    window_min_max : np.ndarray
        Array of shape (num_windows, 2), where each row contains [min, max] values
        used to normalize the corresponding window in X.
    max_appliance_value : float
        Maximum appliance value used to normalize y.
    """

    num_windows = X.shape[0]
    mid_index = num_windows // 2

    print("\n--- Sliding Window Inspection ---")

    print("\nAggregate Windows (X):")
    print("First window:\n", X[0])
    print("Middle window:\n", X[mid_index])
    print("Last window:\n", X[-1])

    print("\nAppliance Values (y):")
    print("First:", y[0])
    print("Middle:", y[mid_index])
    print("Last:", y[-1])

    print("\nWindow Min/Max Values:")
    print("First: min =", window_min_max[0][0], ", max =", window_min_max[0][1])
    print("Middle: min =", window_min_max[mid_index][0], ", max =", window_min_max[mid_index][1])
    print("Last: min =", window_min_max[-1][0], ", max =", window_min_max[-1][1])

    print("\nMax Appliance Value used for normalization:", max_appliance_value)


def plot_mains_and_appliance(df, n_samples=None):
    """
    Plots two time-series graphs showing mains ('aggregate') and appliance power consumption.
    The x-axis represents timestep indices (each corresponding to a 30-second interval).
    The function displays up to the specified number of samples; if not provided, all samples
    from the DataFrame are plotted.

    :param df: pandas.DataFrame containing power data with structure:
               index (DatetimeIndex)
               ├── aggregate (float) – mains power
               └── appliance (float) – appliance power
    :param n_samples: int, optional
                      Number of samples to plot. Defaults to all samples in the DataFrame.
    :return: None
    """

    if n_samples is None or n_samples > len(df):
        n_samples = len(df)

    timesteps = range(n_samples)
    mains = df['aggregate'].iloc[:n_samples]
    appliance = df['appliance'].iloc[:n_samples]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot mains (aggregate)
    axes[0].plot(timesteps, mains, label='Mains (aggregate)', color='tab:blue')
    axes[0].set_ylabel('Power [W]')
    axes[0].set_title('Mains (aggregate)')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Plot appliance
    axes[1].plot(timesteps, appliance, label='Appliance', color='tab:orange')
    axes[1].set_ylabel('Power [W]')
    axes[1].set_xlabel('Timestep (30s intervals)')
    axes[1].set_title('Appliance')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def find_max_appliance_info(df: pd.DataFrame):
    """
    Finds the maximum appliance power value in the given DataFrame and
    returns the corresponding time index and aggregate value.

    :param df: pandas.DataFrame containing columns:
               ├── aggregate (float)
               └── appliance (float)
               indexed by datetime (Time)
    :return: dict with keys:
             {
                 'time': datetime,
                 'appliance_max': float,
                 'aggregate_at_max': float
             }
    """
    if 'appliance' not in df.columns or 'aggregate' not in df.columns:
        raise ValueError("DataFrame must contain 'aggregate' and 'appliance' columns.")

    # Find index (row) where appliance has maximum value
    idx_max = df['appliance'].idxmax()

    # Extract values
    appliance_max = df.loc[idx_max, 'appliance']
    aggregate_at_max = df.loc[idx_max, 'aggregate']

    return {
        'time': idx_max,
        'appliance_max': appliance_max,
        'aggregate_at_max': aggregate_at_max
    }


if __name__ == '__main__':
    data_dict = load_refit_csv_to_memory(
        csv_folder='datasets/test',
        appliance_map_path='datasets/metadata/refit_appliance_map.json')
    mains, appliance = load_downsample_data_from_memory(
        data_dict=data_dict,
        house_id=9,
        appliance_name='fridge',
        start_date='2015-01-01',
        end_date='2015-07-01',
        resample_rate='30S')
    df = combine_and_sync(
        mains=mains,
        appliance=appliance)
    df_agg = pad_data(
        data=df['aggregate'].values,
        window_size=100)

    X, y, agg_min_max, app_max = create_sliding_windows_with_normalization(
        aggregate_padded=df_agg,
        appliance_values=df['appliance'].values,
        window_size=100,
        window_step=1)
    inspect_sliding_window_with_normalization(X, y, agg_min_max, app_max)
    inspect_combined_df(df)






