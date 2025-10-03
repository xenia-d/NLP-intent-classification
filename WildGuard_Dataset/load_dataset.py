import pandas as pd


def load_dataset():

    df_train = pd.read_parquet("hf://datasets/allenai/wildguardmix/train/wildguard_train.parquet")
    df_test = pd.read_parquet("hf://datasets/allenai/wildguardmix/test/wildguard_test.parquet")

    return df_train, df_test



# print(f"Train size: {len(df_train)}")
# print(f"Test size: {len(df_test)}")

# print(df_train.columns)

# print(df_test.columns)