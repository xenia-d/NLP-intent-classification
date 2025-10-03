import torch
from torch.utils.data import DataLoader
from WildGuard_Dataset.WildGuardMixDataset import WildGuardMixDataset
from WildGuard_Dataset.load_dataset import load_dataset






if __name__ == "__main__":

    num_epochs = 20  # Adjust as needed
    batch_size = 32  # Adjust as needed

    df_train, df_val, df_test = load_dataset()

    train_dataset = WildGuardMixDataset(df_train)
    val_dataset = WildGuardMixDataset(df_val)
    test_dataset = WildGuardMixDataset(df_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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

    for bert_model_name in ["DistilBert", "Bert", "Roberta"]:
        print(f"################ Training with {bert_model_name} model...")
        # train_model(bert_model_name, num_epochs=num_epochs, batch_size=batch_size)