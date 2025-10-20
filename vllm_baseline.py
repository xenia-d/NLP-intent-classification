
import pandas as pd
import torch
from vllm import LLM, SamplingParams


def load_dataset(file_path, start_id=1084, end_id=1118):
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

def generate_llm_responses(df, llm, prompt_template, max_new_tokens=50):
    """
    Generate LLM responses for each unique prompt in the dataset.
    """
    llm_responses = []
    sampling_params = SamplingParams(max_tokens=max_new_tokens)

    for _, row in df[['id', 'prompt']].drop_duplicates().iterrows():
        prompt_id = row['id']
        prompt_text = row['prompt']

        full_prompt_text = prompt_template.format(question=prompt_text)
        print(f"Processing prompt ID {prompt_id}")
        outputs = llm.generate([full_prompt_text], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        print("content:", response)
        llm_responses.append({'id': prompt_id, 'LLM_intent': response})
    return pd.DataFrame(llm_responses)


def main():

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    annotations_path = "Annotations/annotations_6.csv"
    prompt_template_path = "Annotations/annotation_guidelines_prompt.txt"

    # Load annotations and and prompt template
    df = load_dataset(annotations_path)
    prompt_template = load_prompt_template(prompt_template_path)

    # Load model and tokenizer
    model_name = 'RedHatAI/Qwen3-8B-quantized.w4a1'
    # tokenizer = AutoTokenizer.from_pretrained(model_name, load_in_4bit=True)
    llm = LLM(model=model_name)

    # Generate LLM responses
    llm_responses_df = generate_llm_responses(df, llm, prompt_template)

    # Merge responses back into the original dataset
    df = df.merge(llm_responses_df, on='id', how='left')
    
    # replace / with _ in model_name
    file_model_name = model_name.replace('/', '_')
    # Save the results
    output_path = f"Annotations/annotations_with_LLM_responses_{file_model_name}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved annotated dataset with LLM responses to {output_path}")


if __name__ == "__main__":
    main()