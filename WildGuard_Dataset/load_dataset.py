import pandas as pd


df_train = pd.read_parquet("hf://datasets/allenai/wildguardmix/train/wildguard_train.parquet")
df_test = pd.read_parquet("hf://datasets/allenai/wildguardmix/test/wildguard_test.parquet")

print(f"Train size: {len(df_train)}")
print(f"Test size: {len(df_test)}")

