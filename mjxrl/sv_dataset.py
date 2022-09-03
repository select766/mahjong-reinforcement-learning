import pickle
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset

class MjxSVDataset(Dataset):
    def __init__(self, path, columns=["obs", "actions"]):
        self.columns = columns
        with gzip.open(path, "rb") as f:
            self.preprocessed = pickle.load(f)

    def __len__(self):
        return len(self.preprocessed["obs"])

    def __getitem__(self, idx):
        values = []
        for column in self.columns:
            if column == "obs":
                # (16, 34)だが(16*34,)に変換する
                values.append(torch.from_numpy(self.preprocessed["obs"][idx].flatten().astype(np.float32)))
            elif column == "actions":
                values.append(int(self.preprocessed["actions"][idx]))
            elif column == "action_masks":
                values.append(torch.from_numpy(self.preprocessed["action_masks"][idx].astype(np.int32)))
            elif column == "discounted_rewards":
                # (1, ), dtype=torch.float32 MSELossを使いやすい形式
                values.append(torch.tensor([self.preprocessed["discounted_rewards"][idx]], dtype=torch.float32))
            else:
                raise NotImplementedError

        return tuple(values)
