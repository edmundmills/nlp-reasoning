from collections import deque
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import wandb

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
    train_size = int(dataset_size*train_test_split)
    test_size = dataset_size - train_size
    print('Tokenizing data...')
    dataset = tokenize_data(data)
    print('Data tokenized.')
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f'{train_size} training samples')
    print(f'{test_size} test samples')
    return train_dataset, test_dataset

def train(model, train_dataset, test_dataset, **params):
    dataloader = DataLoader(train_dataset,
                            batch_size=params['batch_size'],
                            num_workers=4,
                            drop_last=True,
                            sampler=RandomSampler(train_dataset))
    
    optimizer = AdamW(model.parameters(),
                      lr=params['lr'],
                      eps=1e-8)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.train()

    running_eval_len = 5
    running_eval = deque([float('Inf')]*running_eval_len, maxlen=running_eval_len)
    stop = False

    for epoch in range(1, params['epochs'] + 1):
        print(f'Starting Epoch {epoch}')
        total_train_loss = 0

        for step, (input_ids, attention_masks, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids,  
                            attention_mask=attention_masks, 
                            labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss = loss.item()
            wandb.log({'loss': loss})
            total_train_loss += loss

            if step % 50 == 0:
                prev_avg_eval_loss = sum(running_eval) / len(running_eval)
                print(f'Epoch: {epoch}\tStep: {step} \tLoss: {loss:.3f}')
                eval_metrics = eval(model, test_dataset, eval_samples=300, **params)
                print(eval_metrics)
                running_eval.append(eval_metrics['Eval Loss'])
                avg_eval_loss = sum(running_eval) / len(running_eval)
                if avg_eval_loss > prev_avg_eval_loss:
                    stop = 'Eval Loss Increasing, Stopping Early'

            if stop:
                print(stop)
                eval_metrics = eval(model, test_dataset, **params)
                print(eval_metrics)
                return

        avg_train_loss = total_train_loss / len(dataloader)
        print(f'Epoch {epoch}\t\tAvg Training Loss: {avg_train_loss:.3f}')
        eval_metrics = eval(model, test_dataset, **params)
        print(eval_metrics)


def eval(model, test_dataset, eval_samples=None, **params):
    sampler = RandomSampler if eval_samples is not None else SequentialSampler
    dataloader = DataLoader(test_dataset,
                            batch_size=params['batch_size'],
                            num_workers=4,
                            drop_last=True,
                            sampler=sampler(test_dataset))

    eval_samples = eval_samples or len(dataloader)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()

    total_eval_loss = 0
    accuracies = []
    for step, (input_ids, attention_masks, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids,  
                            attention_mask=attention_masks, 
                            labels=labels)
        total_eval_loss += outputs.loss.item()
        logits = outputs.logits.cpu().numpy()
        labels = labels.cpu().numpy().flatten()
        predicted = np.argmax(logits, axis=1).flatten()
        accuracy = np.sum(predicted == labels) / len(labels)
        accuracies.append(accuracy)
        if (step + 1) * params['batch_size'] > eval_samples:
            break

    avg_eval_loss = total_eval_loss / len(accuracies)
    avg_accuracy = sum(accuracies) / len(accuracies)
    metrics = {'Eval Loss': avg_eval_loss, 'Eval Accuracy': avg_accuracy}
    wandb.log(metrics)
    model.train()
    return metrics

if __name__ == '__main__':
    params = {
        'batch_size': 32,
        'lr': 2e-5,
        'epochs': 4,
        'seed': 42,
    }

    wandb.init(
        entity='nlp-reasoning',
        project='sarcasm_detection',
        notes="",
        config=params,
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed_all(params['seed'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_data, test_data = load_data('data/Sarcasm_Headlines_Dataset_v2.json')

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                               num_labels=2,
                                                               output_attentions=False,
                                                               output_hidden_states=False)
    model.to(device)
    
    train(model, train_data, test_data, **params)

