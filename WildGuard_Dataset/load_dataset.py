import pandas as pd


def load_dataset():

    df_train = pd.read_parquet("hf://datasets/allenai/wildguardmix/train/wildguard_train.parquet")
    # df_test = pd.read_parquet("hf://datasets/allenai/wildguardmix/test/wildguard_test.parquet")

    # For quicker testing, a subset of the training data 
    df_train = df_train.iloc[:10000].reset_index(drop=True)

    # divide df_train into train and test sets (80-20 split)
    df_test = df_train.sample(frac=0.2, random_state=12)
    df_train = df_train.drop(df_test.index).reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_train = df_train.dropna(subset=["prompt_harm_label"])
    df_test = df_test.dropna(subset=["prompt_harm_label"])

    print(f"Train dataset size: {len(df_train)}")
    print(f"Test dataset size: {len(df_test)}")

    return df_train, df_test
