import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset():

    df_train = pd.read_parquet("hf://datasets/allenai/wildguardmix/train/wildguard_train.parquet")

    # For quicker testing, a subset of the training data 
    df_train = df_train.iloc[:10000].reset_index(drop=True)

    # divide df_train into train val and test splits (80% train, 10% val, 10% test)
    df_val, df_test = train_test_split(df_train, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42)

    df_train = df_train.dropna(subset=["prompt_harm_label"])
    df_val = df_val.dropna(subset=["prompt_harm_label"])
    df_test = df_test.dropna(subset=["prompt_harm_label"])

    print(f"Train dataset size: {len(df_train)}")
    print(f"Validation dataset size: {len(df_val)}")
    print(f"Test dataset size: {len(df_test)}")

    return df_train, df_val, df_test
