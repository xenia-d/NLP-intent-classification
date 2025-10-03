import pandas as pd


df_train = pd.read_parquet("hf://datasets/allenai/wildguardmix/train/wildguard_train.parquet")
df_test = pd.read_parquet("hf://datasets/allenai/wildguardmix/test/wildguard_test.parquet")

print(f"Train size: {len(df_train)}")
print(f"Test size: {len(df_test)}")

print(df_train.columns)

# remove response-relevant columns and keep only the prompt ones
df_train = df_train.drop(columns=['response', 'response_refusal_label', 'response_harm_label'])
df_test = df_test.drop(columns=['response', 'response_refusal_label', 'response_harm_label'])

print(df_train.columns)