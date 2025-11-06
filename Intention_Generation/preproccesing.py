import json
from datasets import load_dataset, Dataset
import random
import os


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