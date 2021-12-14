from collections import deque
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, AdamW
import wandb

from load_dataset import load_data

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
                return model

        avg_train_loss = total_train_loss / len(dataloader)
        print(f'Epoch {epoch}\t\tAvg Training Loss: {avg_train_loss:.3f}')
        eval_metrics = eval(model, test_dataset, **params)
        print(eval_metrics)
    return model


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

def train_classifier(dataset=None, **params):
    save_dir = f'./models/classifier/{wandb.run.name}'
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not dataset:
        train_data, test_data = load_data('data/Sarcasm_Headlines_Dataset_v2.json')
    else:
        pass

    print('Loading Pretrained Model...')
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                               num_labels=2,
                                                               output_attentions=False,
                                                               output_hidden_states=False)
    model.to(device)
    
    model = train(model, train_data, test_data, **params)
    print(f'Saving model to {save_dir}')
    model.save_pretrained(save_dir)
    return model

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

    train_classifier(**params)


