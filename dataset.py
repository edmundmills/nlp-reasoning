import json
from typing import Dict

import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

class Dataset:
    def __init__(self, file=None):
        if file:
            print('Loading Data...')
            self.data = list(parse_data(file))
            print(f'{len(self)} samples loaded')
            positive = sum(sample['is_sarcastic'] for sample in self.data)
            print(f'{positive/len(self)*100:.1f}% sarcastic samples')
        else:
            self.data = None
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def train_test_split(self, ratio):
        train_size = int(len(self)*ratio)
        test_size = len(self) - train_size
        train_dataset = Dataset()
        train_dataset.data = self.data[:train_size]
        test_dataset = Dataset()
        test_dataset.data = self.data[train_size:]
        print(f'{len(train_dataset)} training samples')
        print(f'{len(test_dataset)} test samples')
        return train_dataset, test_dataset

    def to_tokenized_tensors(self, model:str) -> TensorDataset:
        input_ids = []
        attention_masks = []
        labels = []

        print('Tokenizing data...')

        for sample in self.data:
            encoded_dict = self.tokenizer.encode(sample, model)
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(sample['is_sarcastic'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        tensor_dataset = TensorDataset(input_ids, attention_masks, labels)
        print('Data tokenized.')
        return tensor_dataset


class Tokenizer:
    def __init__(self):
        self.generator_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.classifier_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    def encode(self, sample:Dict, model:str) -> Dict:
        if model == 'generator':
            output = self.generator_tokenizer.encode_plus(sample['headline'],
                                            add_special_tokens = True,
                                            max_length = 200,
                                            padding='max_length',
                                            truncation=True,
                                            return_attention_mask=True,
                                            return_tensors = 'pt') 
        elif model == 'classifier':
            output = self.generator_tokenizer.encode_plus(sample['headline'],
                                            add_special_tokens = True,
                                            max_length = 200,
                                            padding='max_length',
                                            truncation=True,
                                            return_attention_mask=True,
                                            return_tensors = 'pt') 
        else:
            raise ValueError
        return output


