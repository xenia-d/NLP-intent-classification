import torch
from torch.utils.data import DataLoader
from WildGuard_Dataset.WildGuardMixDataset import WildGuardMixDataset
from WildGuard_Dataset.load_dataset import load_dataset


df_train, df_test = load_dataset()

train_dataset = WildGuardMixDataset(df_train)
test_dataset = WildGuardMixDataset(df_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

print(" Training batch example")
for i, batch in enumerate(train_loader):
    print(batch)
    if i >= 1:  # just first 2 batches
        break

print(" Test batch example ")
for i, batch in enumerate(test_loader):
    print(batch)
    if i >= 1:
        break