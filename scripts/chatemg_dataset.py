import os
import pathlib

import numpy as np
import pandas as pd
from scipy.signal import medfilt
from torch.utils.data import Dataset

import misc_utils as mu


class ChatEMGDataset(Dataset):
    def __init__(
        self,
        csv_files,
        filter_class,
        block_size,
        clip_min=0,
        clip_max=999,
        median_filter_size=1,
        shift=False,
        flip=False,
    ):
        self.csv_files = csv_files
        self.filter_class = filter_class
        self.block_size = block_size
        self.shift = shift
        self.flip = flip

        data_list = []
        label_list = []
        print(type(median_filter_size))
        if not isinstance(median_filter_size, int):
            raise Exception("Manual Exception: median_filter_size is not of type int")
        if median_filter_size != 1:
            print(f"Using median filter with size {median_filter_size}")
        for f in self.csv_files:
            data_path = f
            df = pd.read_csv(data_path, index_col=0)
            X, y = mu.clean_dataframe(df)
            X = np.clip(X, a_min=clip_min, a_max=clip_max)
            if median_filter_size != 1:
                X = medfilt(X, kernel_size=[median_filter_size, 1])
            data_list.append(X)
            label_list.append(y)

        # filtering data based on class labels
        self.filtered_data_list = data_list
        if self.filter_class is not None:
            self.filtered_data_list = []
            for d, l in zip(data_list, label_list):
                filtered_d = []
                for i in range(len(d)):
                    if l[i] == self.filter_class:
                        filtered_d.append(d[i])
                        if i + 1 == len(d) or l[i + 1] != self.filter_class:
                            self.filtered_data_list.append(np.array(filtered_d))
                            filtered_d = []
        # now I am removing chunks shorter than block size + 1, because we need to consider y as well
        self.filtered_data_list = [
            d for d in self.filtered_data_list if len(d) >= (self.block_size + 1)
        ]

        # Data augmentation for inter-channel setup
        if self.shift:
            augment_list = []
            for d in self.filtered_data_list:
                for i in range(1, 8):  # shift 7 times
                    d_shifted = np.roll(d, i, axis=-1)
                    augment_list.append(d_shifted)
            for ad in augment_list:
                self.filtered_data_list.append(ad)

        if self.flip:
            augment_list = []
            for d in self.filtered_data_list:
                d_flipped = np.flip(d, axis=-1).copy()
                augment_list.append(d_flipped)
            for ad in augment_list:
                self.filtered_data_list.append(ad)

        # compute mean and std
        self.mean = np.mean(np.concatenate(self.filtered_data_list), axis=0)
        self.std = np.std(np.concatenate(self.filtered_data_list), axis=0)
        self.filtered_data_lens = [
            len(d) - self.block_size for d in self.filtered_data_list
        ]


        # first element is which sublist, second element is which position in the sublist
        self.table = np.zeros((sum(self.filtered_data_lens), 2), dtype=int)
        s = 0
        for i, l in enumerate(self.filtered_data_lens):
            self.table[s : s + l, 0] = i
            self.table[s : s + l, 1] = range(l)
            s = s + l
        print(f"total number of 8-channel samples: {sum(self.filtered_data_lens)}")

    def get_len(self):
        return sum(self.filtered_data_lens)

    def __len__(self):
        return self.get_len()

    def __getitem__(self, item):
        return self.get_sample(item)

    def get_sample(self, item):
        assert item < self.__len__()
        # return (256, 8)
        a, b = self.table[item]
        x = self.filtered_data_list[a][b : b + self.block_size]
        y = self.filtered_data_list[a][b + 1 : b + 1 + self.block_size]
        return x, y

    def sample(self, num):
        idx = np.random.choice(range(self.__len__()), num, replace=False)
        X = []
        Y = []
        for i in idx:
            x, y = self.__getitem__(i)
            X.append(x)
            Y.append(y)
        return np.stack(X), np.stack(Y)


if __name__ == "__main__":
    dataset = ChatEMGDataset(
        csv_files=[
            "2023_02_17_p1/p1_111.csv",
            "2023_02_17_p1/p1_121.csv",
            "2023_02_17_p1/p1_131.csv",
            "2023_02_17_p1/p1_141.csv",
        ],
        filter_class=1,
        block_size=256,
        shift=True,
        flip=True,
    )
    print(f"num samples: {len(dataset)}")
    x, y = dataset[0]
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    print("here")
    print(dataset.table[:10])
    print(len(dataset.filtered_data_list))
