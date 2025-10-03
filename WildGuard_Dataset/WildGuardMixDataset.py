from torch.utils.data import Dataset
import pandas as pd
import torch

class WildGuardMixDataset(Dataset):
    def __init__(self, df: pd.DataFrame, 
                prompt_col="prompt", 
                label_col="prompt_harm_label",
                preprocess_fn=None, #optional
                tokenizer_fn=None): #optional


        # Keep only prompt-relevant columns and remove response-relevant columns
        self.df = df.drop(columns=['response', 'response_refusal_label', 'response_harm_label'])

        self.prompts = self.df[prompt_col].tolist()
        self.labels = self.df[label_col].tolist()

        # encode labels as integers
        self.label2id = {"unharmful": 0, "harmful": 1}
        self.harm_label_ids = [self.label2id[l] for l in self.labels]

        # optional preprocessing and tokenization functions
        self.preprocess_fn = preprocess_fn
        self.tokenizer_fn = tokenizer_fn


    def __len__(self):
        return len(self.prompts)
    

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.harm_label_ids[idx]

        # Apply preprocessing
        if self.preprocess_fn:
            prompt = self.preprocess_fn(prompt)

        # Apply tokenizer
        if self.tokenizer_fn:
            encoding = self.tokenizer_fn(
            prompt,
            truncation=True,
            padding="max_length",  
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),       # remove batch dim
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

        return prompt, label
