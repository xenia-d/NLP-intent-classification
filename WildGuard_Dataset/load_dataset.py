import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset():

    df_train = pd.read_parquet("hf://datasets/allenai/wildguardmix/train/wildguard_train.parquet")

    # For quicker testing, a random subset of the training data 
    full_df = df_train.reset_index(drop=True)
    df_train = full_df.sample(n=10000, random_state=22).reset_index(drop=True)

    # divide df_train into train val and test splits (70% train, 15% val, 15% test)
    df_temp, df_test = train_test_split(df_train, test_size=0.15,random_state=12)
    df_train, df_val = train_test_split(df_temp, test_size=0.1765, random_state=12)

    df_train = df_train.dropna(subset=["prompt_harm_label"])
    df_val = df_val.dropna(subset=["prompt_harm_label"])
    df_test = df_test.dropna(subset=["prompt_harm_label"])

    print(f"Train dataset size: {len(df_train)}")
    print(f"Validation dataset size: {len(df_val)}")
    print(f"Test dataset size: {len(df_test)}")

    print("Label split in train set:", df_train["prompt_harm_label"].value_counts(normalize=True))
    print("Label split in val set:", df_val["prompt_harm_label"].value_counts(normalize=True))
    print("Label split in test set:", df_test["prompt_harm_label"].value_counts(normalize=True))

    return df_train, df_val, df_test
