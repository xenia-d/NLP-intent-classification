import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainingArguments
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


from preproccesing import preprocess_data


def format_input_output(examples, tokenizer, prompt_max=512, intent_max=32):
    inputs = examples["prompt"]
    targets = examples["intent"]

    model_inputs = tokenizer(inputs, max_length=prompt_max, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=intent_max, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["id"] = examples["id"]  # Preserve the original id

    return model_inputs

def get_lengths(dataset, plot=False):
    prompt_lengths = [len(item['prompt'].split()) for item in dataset]
    intent_lengths = [len(item['intent'].split()) for item in dataset]
    prompt_max = max(prompt_lengths)
    intent_max = max(intent_lengths)
    print("Maximum prompt length:", [prompt_max])
    print("Maximum intent length:", [intent_max])
    if plot:
        plt.hist(prompt_lengths, bins=30, alpha=0.5, label='Prompt Lengths')
        plt.hist(intent_lengths, bins=30, alpha=0.5, label='Intent Lengths')
        plt.legend(loc='upper right')
        plt.show()

    return prompt_max, intent_max

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer

def get_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model

def train_val_test_split(dataset, test_size=0.1, val_size=0.1, seed=22):
    # Split the dataset into train, validation, and test set
    train_test_split = dataset.train_test_split(test_size=test_size, seed=seed)
    test_dataset = train_test_split["test"]
    true_val_size = val_size / (1 - test_size)
    val_dataset = train_test_split["train"].train_test_split(test_size=true_val_size, seed=seed)
    train_dataset = val_dataset["train"]
    val_dataset = val_dataset["test"]

    return train_dataset, val_dataset, test_dataset

def t5_trainer(model, tokenizer, train_dataset, val_dataset, intent_max, epochs=8, lr=5e-5, batch_size=8):
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./train_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        generation_max_length=intent_max
    )

    # Define the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    return trainer


def train_model(trainer, tokenizer):
    # Fine-tune the model
    trainer.train()

    # Save the model
    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")


def get_all_preds(model, dataset, tokenizer, device):
    model.eval()

    collate_fn = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    def collate_tensor_fields(examples):
        tensor_inputs = [
            {k: v for k, v in e.items() if k in ["input_ids", "attention_mask", "labels"]}
            for e in examples
        ]
        return collate_fn(tensor_inputs)

    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_tensor_fields)

    predictions = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_ids = torch.argmax(outputs.logits, dim=-1)
        predictions.extend(pred_ids.cpu().numpy())
        torch.cuda.empty_cache()

    # Decode all predictions into text
    decoded_preds = [tokenizer.decode(p, skip_special_tokens=True) for p in predictions]

    return decoded_preds

def evaluate_on_val(trainer, val_dataset):
    # Calculate the evaluation loss
    eval_results = trainer.evaluate(val_dataset)
    print(f"Evaluation Loss: {eval_results['eval_loss']}")


def map_preds_to_intents(predictions, dataset, tokenizer):
    # Map predictions back to original ids
    for i, pred in enumerate(predictions.predictions):
        original_id = dataset[i]["id"]
        pred = np.where(pred != -100, pred, tokenizer.pad_token_id)
        print(f"ID: {original_id}, Prediction: {tokenizer.decode(pred, skip_special_tokens=True)}")

def save_preds(model_name, predictions, eval_dataset, tokenizer):
    # Make predictions directory if it does not exist
    if not os.path.exists('predictions'):
        os.makedirs('predictions')

    # save the predictions to a json file
    with open('predictions/{}.json'.format(model_name.replace('/', '_')), 'w', encoding='utf-8') as f:
        for i, pred in enumerate(predictions.predictions):
            original_id = eval_dataset[i]["id"]
            pred = np.where(pred != -100, pred, tokenizer.pad_token_id)
            decoded_pred = tokenizer.decode(pred, skip_special_tokens=True)
            true_intent = eval_dataset[i]["intent"]
            json.dump({'id': original_id, 'prediction': decoded_pred, 'true_intent': true_intent}, f, ensure_ascii=False)
            f.write('\n')

def train_main(model_name="t5-small"):
    final_dataset = preprocess_data()
    prompt_max, intent_max = get_lengths(final_dataset, plot=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name=model_name).to(device)
    tokenizer = get_tokenizer(model_name=model_name)

    tokenized_dataset = final_dataset.map(format_input_output, fn_kwargs={"tokenizer": tokenizer, "prompt_max": prompt_max, "intent_max": intent_max}, batched=True)
    train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_dataset, test_size=0.1, val_size=0.1, seed=22)

    trainer = t5_trainer(model, tokenizer, train_dataset, val_dataset, intent_max)
    train_model(trainer, tokenizer)

    evaluate_on_val(trainer, val_dataset)
    preds_val = trainer.predict(val_dataset)
    save_preds(model_name, preds_val, val_dataset, tokenizer)
    preds_test = trainer.predict(test_dataset)
    save_preds(model_name, preds_test, test_dataset, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for intent classification.")
    parser.add_argument("--model_name", type=str, default="t5-small", help="Name of the pre-trained model to use.")
    args = parser.parse_args()

    train_main(model_name=args.model_name)
