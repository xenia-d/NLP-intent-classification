import json
from datasets import load_dataset
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import os
import argparse
from tqdm import tqdm
from peft import PeftModel

def load_model(model_name="t5-small"):
    full_model_name = "trained_models/" + model_name + "-model"

    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    if "t5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(full_model_name)
    if "qwen" in model_name.lower():
        if "lora" in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_8bit=True)
            model = PeftModel.from_pretrained(model, full_model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=True)

    return model, tokenizer


def get_dataset(num_samples=20):
    dataset = load_dataset("AmazonScience/FalseReject", split="train")
    prompts = dataset['prompt']

    random.seed(22)
    prompts = random.sample(prompts, num_samples) # Get a random sample of prompts

    return prompts

def run_inference(model, tokenizer, inputs, model_name="t5-small"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "lora" not in model_name.lower():
        model.to(device)
    model.eval()

    generated_intents = []
    instruction_text = "Generate the intent for the following text: "

    for prompt in tqdm(inputs):
        if "t5" in model_name:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(input_ids, max_length=32, num_return_sequences=1)

        if "qwen" in model_name.lower():
            prompt = f"<|user|>\n{instruction_text}{prompt}\n<|assistant|>\n"
            input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)
            
            with torch.no_grad():
                outputs = model.generate(input_ids, attention_mask=attention_mask, num_return_sequences=1, max_new_tokens=32)

        intent = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_intents.append(intent)

    return generated_intents

def save_prompts(prompts):
    folder = "generated_intents"
    filename = "prompts_AmazonScience.json"
    full_filename = os.path.join(folder, filename)
    with open(full_filename, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=4)

    print(f"Prompts saved to {full_filename}")

def save_intents(intents, prompts, model_name):
    folder = "generated_intents"
    filename = model_name + "_AmazonScience_intents.json"
    full_filename = os.path.join(folder, filename)
    with open(full_filename, 'w', encoding='utf-8') as f:
        intent_data = [{"prompt": prompt, "generated_intent": intent} for prompt, intent in zip(prompts, intents)]
        json.dump(intent_data, f, ensure_ascii=False, indent=4)

    print(f"Generated intents saved to {full_filename}")


def main(model_name="t5-small"):
    prompts = get_dataset(num_samples=20)
    # save_prompts(prompts)
    model, tokenizer = load_model(model_name=model_name)
    generated_intents = run_inference(model, tokenizer, prompts, model_name=model_name)
    save_intents(generated_intents, prompts, model_name=model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intent Generation using LMs")
    parser.add_argument('--model_name', type=str, default='t5-small', help='Name of the model to use')
    args = parser.parse_args()

    main(model_name=args.model_name)