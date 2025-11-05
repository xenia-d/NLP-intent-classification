import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader
from WildGuard_Dataset.WildGuardMixDataset import WildGuardMixDataset
from WildGuard_Dataset.load_dataset import load_dataset
from Model.model import get_model, get_tokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch.nn as nn
import os

def train_model(model_name, train_loader, val_loader, device, num_epochs=5, lr=1e-5, save_path="Saved_Models"):
    # Create model for this backbone
    model = get_model(device, model_name)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # -------------------------
        # Training
        # -------------------------
        model.train()
        total_loss = 0
        all_train_preds, all_train_labels = [], []

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds, average="macro")

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds, average="macro")

        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_file = f"{save_path}/{model_name}_best.pt"
            torch.save(model.state_dict(), save_file)
            print(f"Saved best {model_name} model to {save_file}")

    print(f"\n=== Finished Training {model_name} ===")
    print(f"Best Val F1: {best_val_f1:.4f}")




if __name__ == "__main__":

    num_epochs = 3  # Adjust as needed
    batch_size = 16 # 32  # Adjust as needed
    lr = 1e-5        # Adjust as needed
    save_path = "Saved_Models"


    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    df_train, df_val, df_test = load_dataset()
    curr_model_iteration = 0

    for bert_model_name in ["Roberta"]: # DistilBert, "Bert", "Roberta"
        while curr_model_iteration < 5:

            while os.path.exists(f"Saved_Models/{bert_model_name}/iteration_{curr_model_iteration}"):
                print(f"Model path {bert_model_name}/iteration_{curr_model_iteration} already exists. Incrementing iteration.")
                curr_model_iteration += 1
            model_save_path = f"Saved_Models/{bert_model_name}/iteration_{curr_model_iteration}"
            os.makedirs(model_save_path)

            tokenizer = get_tokenizer(bert_model_name)

            train_dataset = WildGuardMixDataset(df_train, tokenizer_fn=tokenizer)
            val_dataset = WildGuardMixDataset(df_val, tokenizer_fn=tokenizer)
            test_dataset = WildGuardMixDataset(df_test, tokenizer_fn=tokenizer)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)


            print(f"################ Training with {bert_model_name} model...")
            train_model(bert_model_name, train_loader, val_loader, device, num_epochs, lr, model_save_path)

