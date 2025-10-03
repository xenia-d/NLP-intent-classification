import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from WildGuard_Dataset.WildGuardMixDataset import WildGuardMixDataset
from WildGuard_Dataset.load_dataset import load_dataset


df_train, df_test = load_dataset()

print(f"Train size: {len(df_train)}")
print(f"Test size: {len(df_test)}")