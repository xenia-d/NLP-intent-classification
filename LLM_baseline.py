import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import scipy.stats as stats
import os

def load_dataset(file_path, start_id=1084, end_id=1088):
    """
    Load and filter the dataset by including a range of prompt IDs and keep relevant columns.
    """
    df = pd.read_csv(file_path)
    df = df[(df['id'] >= start_id) & (df['id'] <= end_id)]
    df = df[['id', 'prompt', 'annotator', 'intent']]
    return df

def load_prompt_template(template_path):
    """
    Load the prompt template from a text file.
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    return template

def generate_llm_responses(df, tokenizer, model, prompt_template, max_new_tokens=50):
    """
    Generate LLM responses for each unique prompt in the dataset.
    """
    llm_responses = []

    for _, row in df[['id', 'prompt']].drop_duplicates().iterrows():
        prompt_id = row['id']
        prompt_text = row['prompt']

        # Fill in the system prompt template
        full_prompt_text = prompt_template.format(question=prompt_text)

        # Apply the chat template (disable thinking)
        full_prompt = tokenizer.apply_chat_template(
            full_prompt_text,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        print(f"Processing prompt ID {prompt_id}")

        # Tokenize and generate response
        inputs = tokenizer(full_prompt, return_tensors='pt')
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        print("If this prints then thats great news!! ")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        llm_responses.append({'id': prompt_id, 'LLM_intent': response})

    return pd.DataFrame(llm_responses)


def main():

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    annotations_path = "Annotations/annotations_5.csv"
    prompt_template_path = "Annotations/annotation_guidelines_prompt.txt"

    # Load annotations and and prompt template
    df = load_dataset(annotations_path)
    prompt_template = load_prompt_template(prompt_template_path)

    # Load model and tokenizer
    model_name = 'Qwen/Qwen3-8B'
    tokenizer = AutoTokenizer.from_pretrained(model_name, load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

    # Generate LLM responses
    llm_responses_df = generate_llm_responses(df, tokenizer, model, prompt_template)

    # Merge responses back into the original dataset
    df = df.merge(llm_responses_df, on='id', how='left')

    # Save the results
    output_path = "Annotations/annotations_with_LLM_responses.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved annotated dataset with LLM responses to {output_path}")


if __name__ == "__main__":
    main()