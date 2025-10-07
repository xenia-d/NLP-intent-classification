import torch
from torch.utils.data import DataLoader
from WildGuard_Dataset.WildGuardMixDataset import WildGuardMixDataset
from WildGuard_Dataset.load_dataset import load_dataset
from Model.model import get_model, get_tokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch.nn as nn
import os
import pandas as pd
import numpy as np

def get_final_preds(logits_dict):
    # Convert logits to numpy arrays for easier manipulation
    logits_arrays = {k: np.array(v) for k, v in logits_dict.items()}
    
    # Stack logits from different models
    stacked_logits = np.stack(list(logits_arrays.values()), axis=0)  # Shape: (num_models, num_samples, num_classes)
    
    # Average the logits across models
    avg_logits = np.mean(stacked_logits, axis=0)  # Shape: (num_samples, num_classes)
    
    # Get final predictions by taking the argmax of the averaged logits
    final_preds = np.argmax(avg_logits, axis=1)
    
    return final_preds

def get_entropy(logits_dict):
    # Convert logits to numpy arrays for easier manipulation
    logits_arrays = {k: np.array(v) for k, v in logits_dict.items()}
    
    # Stack logits from different models
    stacked_logits = np.stack(list(logits_arrays.values()), axis=0)  # Shape: (num_models, num_samples, num_classes)
    
    # Average the logits across models
    avg_logits = np.mean(stacked_logits, axis=0)  # Shape: (num_samples, num_classes)
    
    # Convert averaged logits to probabilities using softmax
    exp_logits = np.exp(avg_logits - np.max(avg_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # Shape: (num_samples, num_classes)
    
    # Calculate entropy for each sample
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)  # Shape: (num_samples,)
    
    return entropy




if __name__ == "__main__":
    df_train, df_val, df_test = load_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    models = []

    bert_model_name = "DistilBert"
    model_path = f'Saved_Models/{bert_model_name}'
    batch_size = 64  # Adjust as needed

    # loop through all folders in directory
    for folder_name in os.listdir(model_path):
        folder_path = os.path.join(model_path, folder_name)
        if os.path.isdir(folder_path):
            model_file = os.path.join(folder_path, f'{bert_model_name}_best.pt')
            if os.path.exists(model_file):
                model = get_model(device, bert_model_name)
                model.load_state_dict(torch.load(model_file, map_location=device))
                model.eval()
                models.append(model)
                print(f"Loaded model from {folder_name}")

    tokenizer = get_tokenizer(bert_model_name)
    test_dataset = WildGuardMixDataset(df_test, tokenizer_fn=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    all_logits = {}
    all_labels = []
    all_prompts = []

    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        for i, model in enumerate(models):

            # initialize list to store logits for each model
            if all_logits.get(i) is None:
                all_logits[i] = []

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # append logits to corresponding list
            all_logits[i].extend(logits.detach().cpu().numpy())

        # save labels to get accuracy and f1 score
        all_labels.extend(labels.cpu().numpy())
        all_prompts.extend(batch["prompts"])

    all_preds = get_final_preds(all_logits)
    all_entropies = get_entropy(all_logits)

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
    print(len(all_entropies), len(all_labels), len(all_prompts), len(all_preds))
    
    df = pd.DataFrame({
        "prompts": all_prompts,
        "labels": all_labels,
        "predictions": all_preds,
        "entropies": all_entropies
    })

    for key in all_logits.keys():
        df[f'logits_model_{key}'] = all_logits[key]
        
    # save dataframe to csv
    df.to_csv(f"ensembled_results_{bert_model_name}.csv", index=False)
