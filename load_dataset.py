import json
from typing import Dict, List, Tuple

import torch
from torch.utils.data import TensorDataset, random_split
from transformers import AutoTokenizer

def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

def tokenize(sample:Dict) -> Dict:
    return tokenizer.encode_plus(sample['headline'],
                                    add_special_tokens = True,
                                    max_length = 200,
                                    padding='max_length',
                                    truncation=True,
                                    return_attention_mask=True,
                                    return_tensors = 'pt') 

def tokenize_data(data:List[Dict]) -> TensorDataset:
    input_ids = []
    attention_masks = []
    labels = []

    for sample in data:
        encoded_dict = tokenize(sample)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(sample['is_sarcastic'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

def load_data(file, train_test_split=0.8) -> Tuple[TensorDataset, TensorDataset]:
    print('Loading Data...')
    data = list(parse_data(file))
    dataset_size = len(data)
    print(f'{dataset_size} samples loaded')
    positive = sum(sample['is_sarcastic'] for sample in data)
    print(f'{positive/dataset_size*100:.1f}% sarcastic samples')
    train_size = int(dataset_size*train_test_split)
    test_size = dataset_size - train_size
    print('Tokenizing data...')
    dataset = tokenize_data(data)
    print('Data tokenized.')
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f'{train_size} training samples')
    print(f'{test_size} test samples')
    return train_dataset, test_dataset