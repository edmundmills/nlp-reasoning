import json
import os
from typing import Dict

import re
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

from utils import download_winning_args

def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

def clean_text(text):
    if text:
        return re.sub(r"(@\[A-Za-z0-9]+)|(\w+:\/\/\S+)|http.+?", "", text)
    else:
        return text

class Dataset:
    def __init__(self, dataset=None):
        self.dataset = dataset
        if dataset is not None:
            print('Loading Data...')
            self.data = self._load_dataset(dataset)
            print(f'{len(self)} samples loaded')
        else:
            self.data = None
        self.tokenizer = Tokenizer()

    def _load_dataset(self, dataset):
        if dataset  == 'sarcasm_headlines':
            data = list(parse_data('data/Sarcasm_Headlines_Dataset_v2.json'))
            positive = sum(sample['is_sarcastic'] for sample in self.data)
            print(f'{positive/len(self)*100:.1f}% sarcastic samples')
        elif dataset == 'winning_arguments':
            data_dict = {}
            data = []
            print('Building reply lookup dict...')
            if not os.path.isdir('data/winning-args-corpus'):
                download_winning_args()
            for row in parse_data('data/winning-args-corpus/utterances.json'):
                text = clean_text(row['text'])
                if text and len(text) > 25 and len(text) < 500:
                    data_dict[row['id']] = text
            print('Building dataset...')
            for row in parse_data('data/winning-args-corpus/utterances.json'):
                prompt_id = row['reply-to']
                prompt = data_dict.get(prompt_id, None)
                response = data_dict.get(row['id'], None)
                success = row['meta']['success']
                score = row['meta']['score'] or 0
                if (success or score >= 5) and prompt and response:
                    data.append({
                        'prompt': prompt,
                        'response': response,
                    })
        else:
            raise ValueError
        return data

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


