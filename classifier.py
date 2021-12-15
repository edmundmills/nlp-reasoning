from collections import deque
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW
import wandb


class Classifier:
    def __init__(self, pretrain_args=None, pretrain_dataset=None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if pretrain_dataset is None:
            self.load_classifier()
        else:
            self.fine_tune(pretrain_dataset, pretrain_args)
        self.model = self.model.to(self.device)

    def load_classifier(self):
        classifiers = Path('models/classifier')
        classifier_dirs = [dir for dir in classifiers.iterdir() if dir.is_dir()]
        classifier_dirs = sorted(classifier_dirs, key=lambda t: -os.stat(t).st_mtime)
        self.model_dir = classifier_dirs[0]
        print(f'Loading classifier from {str(self.model_dir)}')
        self.model = BertForSequenceClassification.from_pretrained(self.model_dir)

    def fine_tune(self, dataset, args):
        if not args.debug:
            self.model_dir = f'./models/classifier/{wandb.run.name}'
            os.makedirs(self.model_dir, exist_ok=True)

        train_data, test_data = dataset.train_test_split(0.8)
        train_data = train_data.to_tokenized_tensors(model='classifier')
        test_data = test_data.to_tokenized_tensors(model='classifier')

        print('Loading Pretrained Model...')
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                num_labels=2,
                                                                output_attentions=False,
                                                                output_hidden_states=False)
        self.model.to(self.device)
        
        model = self.train(train_data, test_data, args)
        if not args.debug:
            print(f'Saving model to {self.model_dir}')
            model.save_pretrained(self.model_dir)
        return model


    def train(self, train_dataset, test_dataset, args):
        dataloader = DataLoader(train_dataset,
                                batch_size=args.classifier_training_batch_size,
                                num_workers=4,
                                drop_last=True,
                                sampler=RandomSampler(train_dataset))
        
        optimizer = AdamW(self.model.parameters(),
                        lr=args.classifier_training_lr,
                        eps=1e-8)

        self.model.train()

        running_eval_len = 5
        running_eval = deque([float('Inf')]*running_eval_len, maxlen=running_eval_len)
        stop = False

        for epoch in range(1, args.classifier_training_epochs + 1):
            print(f'Starting Epoch {epoch}')
            total_train_loss = 0

            for step, (input_ids, attention_masks, labels) in enumerate(dataloader):
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(input_ids,  
                                     attention_mask=attention_masks, 
                                     labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                loss = loss.item()
                wandb.log({'loss': loss})
                total_train_loss += loss

                if step % 50 == 0:
                    prev_avg_eval_loss = sum(running_eval) / len(running_eval)
                    print(f'Epoch: {epoch}\tStep: {step} \tLoss: {loss:.3f}')
                    eval_metrics = self.eval(test_dataset, args, eval_samples=300)
                    print(eval_metrics)
                    running_eval.append(eval_metrics['Eval Loss'])
                    avg_eval_loss = sum(running_eval) / len(running_eval)
                    if avg_eval_loss > prev_avg_eval_loss:
                        stop = 'Eval Loss Increasing, Stopping Early'

                if stop:
                    print(stop)
                    eval_metrics = self.eval(test_dataset, args)
                    print(eval_metrics)
                    return

            avg_train_loss = total_train_loss / len(dataloader)
            print(f'Epoch {epoch}\t\tAvg Training Loss: {avg_train_loss:.3f}')
            eval_metrics = self.eval(test_dataset, args)
            print(eval_metrics)

    def eval(self, test_dataset, args, eval_samples=None):
        sampler = RandomSampler if eval_samples is not None else SequentialSampler
        dataloader = DataLoader(test_dataset,
                                batch_size=args.classifier_training_batch_size,
                                num_workers=4,
                                drop_last=True,
                                sampler=sampler(test_dataset))

        eval_samples = eval_samples or len(dataloader)

        self.model.eval()

        total_eval_loss = 0
        accuracies = []
        for step, (input_ids, attention_masks, labels) in enumerate(dataloader):
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids,  
                                     attention_mask=attention_masks, 
                                     labels=labels)
            total_eval_loss += outputs.loss.item()
            logits = outputs.logits.cpu().numpy()
            labels = labels.cpu().numpy().flatten()
            predicted = np.argmax(logits, axis=1).flatten()
            accuracy = np.sum(predicted == labels) / len(labels)
            accuracies.append(accuracy)
            if (step + 1) * args.classifier_training_batch_size > eval_samples:
                break

        avg_eval_loss = total_eval_loss / len(accuracies)
        avg_accuracy = sum(accuracies) / len(accuracies)
        metrics = {'Eval Loss': avg_eval_loss, 'Eval Accuracy': avg_accuracy}
        wandb.log(metrics)
        self.model.train()
        return metrics


