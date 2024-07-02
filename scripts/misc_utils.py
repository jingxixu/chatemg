import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import medfilt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.font_manager

from_int_to_str = {0: "relax", 1: "open", 2: "close"}

from_str_to_int = {"relax": 0, "open": 1, "close": 2}

EMG_COLORS = ["b", "g", "r", "c", "m", "y", "k", "tab:brown"]


def plot_emg_chunks_parallel(
    real_data,
    synthetic_data,
    nrows=1,
    ncols=1,
    vertical_location=None,
    flatten=False,
    rmse=None,
    save_fnm=None,
):
    """
    new plotting function that puts real and synthetic data in parallel.

    real_data: (b, t, 8)
    synthetic_data: (b, t, 8)
    b = nrows * ncols
    rmse: (b, )
    """

    # putting real and synthetic in parallel
    nrows = nrows * 2

    # missing the batch dimension
    if np.ndim(real_data) == 2 and not flatten:
        real_data = real_data[None, ...]
    b, t = real_data.shape[:2]
    if flatten:
        real_data = real_data.reshape(b, -1, 8)
        t = real_data.shape[1]
    fig, axs = plt.subplots(figsize=(15, 12), nrows=nrows, ncols=ncols, squeeze=False)

    idx = np.array(range(t)) / 100.0
    colors = ["b", "g", "r", "c", "m", "y", "k", "tab:brown"]

    for i in range(nrows):
        for j in range(ncols):
            for c in range(8):
                if i % 2 == 0:
                    y = real_data[i // 2 * ncols + j, :, c]
                    axs[i, j].set_ylabel("real")
                else:
                    y = synthetic_data[i // 2 * ncols + j, :, c]
                    axs[i, j].set_ylabel("synthetic")
                    axs[i, j].set_xlabel(f"rmse: {rmse[i // 2 * ncols + j]:.2f}")
                axs[i, j].plot(idx, y, label=f"emg{c}", alpha=0.7, color=colors[c])
                if vertical_location is not None:
                    axs[i, j].axvline(x=vertical_location / 100.0, c="b")

    # if 'motor_position' in df.columns:
    #     plt.plot(idx, motor_position, 'tab:grey', label='motor')
    # plt.plot(idx, gt, 'tab:orange', label='gt')

    plt.tight_layout()
    # only plot the legend on the last plot
    # handles, labels = axs[-1, -1].get_legend_handles_labels()
    # fig.legend(handles, labels)
    if save_fnm is not None:
        plt.savefig(f"{save_fnm}", dpi=300, bbox_inches="tight")
    plt.show()


def get_batch(
    split, train_data_list, test_data_list, batch_size, block_size, device_type, device
):
    data_list = train_data_list if split == "train" else test_data_list
    num_per_list = np.round(
        np.array([len(a) for a in data_list])
        / sum([len(a) for a in data_list])
        * batch_size
    ).astype(int)
    num_per_list[-1] = batch_size - sum(num_per_list[:-1])
    x = []
    y = []
    for idx, num in enumerate(num_per_list):
        ix = torch.randint(len(data_list[idx]) - block_size, (num,))
        # TODO normalizing the emg signals???
        x = x + [
            torch.from_numpy((data_list[idx][i : i + block_size]).astype(np.float32))
            for i in ix
        ]
        y = y + [
            torch.from_numpy(
                (data_list[idx][i + 1 : i + 1 + block_size]).astype(np.int64)
            )
            for i in ix
        ]
    x = torch.stack(x)
    y = torch.stack(y)
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def clean_dataframe(df):
    X_df = keep_columns(df, [f"emg"])
    X = X_df.to_numpy()
    y_df = keep_columns(df, ["gt"])
    y = y_df.to_numpy().squeeze()
    return X, y


def keep_columns(df, tuple_of_columns):
    """
    Given a dataframe, and a tuple of column names, this function will search
    through the dataframe and keep only columns which contain a string from the
    list of the desired columns. All other columns are removed
    """
    if len(tuple_of_columns) >= 1:
        cols = df.columns[
            df.columns.to_series().str.contains("|".join(tuple_of_columns))
        ]
        return df[cols]
    return df


def medfilt_wo_padding(arr, window_size):
    """
    built on scipy medial filter which filters using the current element as center and with 0 paddings.
    This function has no padding, so the length will be shortened.

    arr: (b, t, c)
    window_size must be odd
    """
    if window_size == 1:
        return arr
    s = int((window_size - 1) / 2)
    e = -s
    return medfilt(arr, [1, window_size, 1])[:, s:e, :]


def sample_from_dataset(dataset, num, replace=False):
    # sample some samples without replacement
    # This sample 8-channel signals not separate channel signals
    idx = np.random.choice(range(len(dataset)), num, replace=replace)
    X = []
    Y = []
    for i in idx:
        x, y = dataset[i]
        X.append(x)
        Y.append(y)
    return np.stack(X), np.stack(Y)


def compute_mse(y, y_hat, starting_pos):
    """
    y: np array (n, 256, 8)
    y_hat: np array (n, 256, 8)
    return: np array (n, )
    """
    return ((y[:, starting_pos:, :] - y_hat[:, starting_pos:, :]) ** 2).mean(
        axis=(1, 2)
    )
