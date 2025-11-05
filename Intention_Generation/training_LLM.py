import json
from datasets import load_dataset, Dataset
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainingArguments
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from trl import SFTTrainer


def load_dataset():
    current_dir = os.path.dirname(__file__)
    annotations_file_path = os.path.join(current_dir, '..', 'WildGuard_Dataset', 'our_annotations.json')

    with open(os.path.abspath(annotations_file_path), 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    dataset = Dataset.from_list(data_json)
    dataset = dataset.remove_columns(['file_upload', 'drafts', 'predictions', 'meta', 'created_at', 'updated_at', 'inner_id', 'cancelled_annotations', 'total_predictions', 'comment_count', 'unresolved_comment_count', 'last_comment_updated_at', 'project', 'updated_by', 'comment_authors'])

    return dataset

def make_duplicates_seperate(dataset):
    counter = 0
    total_annotations_expanded = 0
    expanded_data = []

    for item in dataset:
        if len(item['annotations']) > 1:
            counter += 1
            total_annotations_expanded += len(item['annotations'])
            # replace annotations with a random one of the annotations
            for annotation in item['annotations']:
                new_item = item.copy()
                new_item['annotations'] = [annotation]
                expanded_data.append(new_item)
        else:
            expanded_data.append(item)

    print(f"Expanded {counter} items with multiple annotations.")
    print(f"Total annotations expanded: {total_annotations_expanded}")
    print(f"Original dataset size: {len(dataset)}")
    print(f"New dataset size: {len(expanded_data)}")

    # Create a new dataset with the modified data
    new_dataset = Dataset.from_list(expanded_data)

    return new_dataset

def select_random_duplicate(dataset):
    counter = 0
    modified_data = []

    for item in dataset:
        if len(item['annotations']) > 1:
            counter += 1
            # replace annotations with a random one of the annotations
            rand_index = random.randint(0, len(item['annotations']) - 1)
            item['annotations'] = [item['annotations'][rand_index]]
        modified_data.append(item)

    print(f'There are {counter} items with more than one annotation. A random annotation was chosen as the final annotation')

    # Create a new dataset with the modified data
    new_dataset = Dataset.from_list(modified_data)

    return new_dataset

def clean_incomplete_data(dataset):
    counter = 0
    no_intent_label = 0

    reduced_data = []
    for item in dataset:
        annotation = item['annotations'][0]['result']
        if len(annotation) < 2 :
            if annotation[0]['from_name'] == 'harmful_class':
                # if there is only the harm label, we do not keep the item as it is missing intention
                no_intent_label += 1
                continue
            elif annotation[0]['from_name'] != 'harmful_class':
                # if there is only the intention label, we keep the item
                reduced_data.append(item)
        else:
            remove_item = False
            for res in annotation:
                if res['from_name'] == 'harmful_class' and res['value']['choices'][0] == 'Flag for Removal':
                    remove_item = True
                    counter += 1
            if not remove_item:
                reduced_data.append(item)
                
    print("flagged for removal:", counter)
    print("no intent label:", no_intent_label)
    print(f'Original dataset size: {len(dataset)}')
    print('Reduced dataset size:', len(reduced_data))
    reduced_data = Dataset.from_list(reduced_data)

    return reduced_data

def get_prompt_and_intents(dataset):
    #  prepare the dataset for trainig by extracting the prompt in 'data', the intent, and the id
    final_data = []
    for item in dataset:
        prompt = item['data']['prompt']
        intent = item['annotations'][0]['result']
        intent_label = None
        for res in intent:
            if res['from_name'] == 'intent':
                intent_label = res['value']['text'][0]
        final_data.append({
            'id': item['id'],
            'prompt': prompt,
            'intent': intent_label
        })
    final_dataset = Dataset.from_list(final_data)

    return final_dataset

def preprocess_data():
    dataset = load_dataset()
    dataset = make_duplicates_seperate(dataset)
    dataset = clean_incomplete_data(dataset)
    final_dataset = get_prompt_and_intents(dataset)

    return final_dataset
    

def preprocess_t5(examples, tokenizer, prompt_max=512, intent_max=32):
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

def evaluate_on_val(trainer, val_dataset):
    # Calculate the evaluation loss
    eval_results = trainer.evaluate(val_dataset)
    print(f"Evaluation Loss: {eval_results['eval_loss']}")

    # Generate predictions
    predictions = trainer.predict(val_dataset)

    return predictions

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
    with open('predictions/{}.json'.format(model_name), 'w', encoding='utf-8') as f:
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

    tokenized_dataset = final_dataset.map(preprocess_t5, fn_kwargs={"tokenizer": tokenizer, "prompt_max": prompt_max, "intent_max": intent_max}, batched=True)
    train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_dataset, test_size=0.1, val_size=0.1, seed=22)

    trainer = t5_trainer(model, tokenizer, train_dataset, val_dataset, intent_max)
    train_model(trainer, tokenizer)

    preds = evaluate_on_val(trainer, val_dataset)
    save_preds(model_name, preds, val_dataset, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for intent classification.")
    parser.add_argument("--model_name", type=str, default="t5-small", help="Name of the pre-trained model to use.")
    args = parser.parse_args()

    train_main(model_name=args.model_name)
