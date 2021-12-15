from collections import deque
import itertools
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AdamW, AutoTokenizer
import wandb

from nlp_reasoning.dataset import ReasoningSample, ReplayBuffer
from nlp_reasoning.model import Model

class Tokenizer:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    def encode(self, sample:Dict) -> Dict:
        prompt = sample['prompt']
        response = sample['response']
        output = self.tokenizer.encode_plus(text=prompt,
                                            text_pair=response,
                                            add_special_tokens = True,
                                            max_length = 400,
                                            padding='max_length',
                                            truncation=True,
                                            return_attention_mask=True,
                                            return_tensors = 'pt') 
        return output

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)
    
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


class Generator(Model):
    def __init__(self) -> None:
        super().__init__()
        self.model_class = GPTNeoForCausalLM
        print('Loading Pretrained Model...')
        self.model = self.model_class.from_pretrained("EleutherAI/gpt-neo-1.3B")
        self.model.to(self.device)
        self.tokenizer = Tokenizer()

    def generate(self, prompts:List[str]) -> List[str]:
        responses = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            gen_tokens = self.model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100)
            gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
            responses.append(gen_text)
        return responses

    def fine_tune(self, dataset, args):
        if not args.debug:
            self.model_dir = f'./models/generator/{wandb.run.name}'

        train_data, test_data = dataset.train_test_split(0.8)
        train_data = train_data.to_tokenized_tensors(tokenizer=self.tokenizer)
        test_data = test_data.to_tokenized_tensors(tokenizer=self.tokenizer)
        
        self.train(train_data, test_data, args)

        if not args.debug:
            self.save()


    # def train(self, train_dataset, test_dataset, args):
    #     dataloader = DataLoader(train_dataset,
    #                             batch_size=args.classifier_training_batch_size,
    #                             num_workers=4,
    #                             drop_last=True,
    #                             sampler=RandomSampler(train_dataset))
        
    #     optimizer = AdamW(self.model.parameters(),
    #                     lr=args.classifier_training_lr,
    #                     eps=1e-8)

    #     self.model.train()

    #     running_eval_len = 5
    #     running_eval = deque([float('Inf')]*running_eval_len, maxlen=running_eval_len)
    #     stop = False

    #     for epoch in range(1, args.classifier_training_epochs + 1):
    #         print(f'Starting Epoch {epoch}')
    #         total_train_loss = 0

    #         for step, (input_ids, attention_masks, labels) in enumerate(dataloader):
    #             input_ids = input_ids.to(self.device)
    #             attention_masks = attention_masks.to(self.device)
    #             labels = labels.to(self.device)
                
    #             outputs = self.model(input_ids,  
    #                                  attention_mask=attention_masks, 
    #                                  labels=labels)
    #             loss = outputs.loss

    #             optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    #             optimizer.step()

    #             loss = loss.item()
    #             wandb.log({'loss': loss})
    #             total_train_loss += loss

    #             if step % 50 == 0:
    #                 prev_avg_eval_loss = sum(running_eval) / len(running_eval)
    #                 print(f'Epoch: {epoch}\tStep: {step} \tLoss: {loss:.3f}')
    #                 eval_metrics = self.eval(test_dataset, args, eval_samples=300)
    #                 print(eval_metrics)
    #                 running_eval.append(eval_metrics['Eval Loss'])
    #                 avg_eval_loss = sum(running_eval) / len(running_eval)
    #                 if avg_eval_loss > prev_avg_eval_loss:
    #                     stop = 'Eval Loss Increasing, Stopping Early'

    #             if stop:
    #                 print(stop)
    #                 eval_metrics = self.eval(test_dataset, args)
    #                 print(eval_metrics)
    #                 return

    #         avg_train_loss = total_train_loss / len(dataloader)
    #         print(f'Epoch {epoch}\t\tAvg Training Loss: {avg_train_loss:.3f}')
    #         eval_metrics = self.eval(test_dataset, args)
    #         print(eval_metrics)      


    # def train_reasoner(dataset, classifier, args):

        # dataset = Dataset('sarcasm_headlines')
        # train_dataset, test_dataset = dataset.train_test_split(0.8)

        # replay_buffer = ReplayBuffer()

        # generator = BertForNextSentencePrediction.from_pretrained("bert-base-uncased",
        #                                                         output_attentions=False,
        #                                                         output_hidden_states=False)
        # generator = generator.to(device)

        # optimizer = AdamW(generator.parameters(),
        #                 lr=params['generator_lr'],
        #                 eps=1e-8)

        # dataloader = DataLoader(train_dataset,
        #                         batch_size=params['generator'],
        #                         num_workers=4,
        #                         drop_last=True,
        #                         sampler=RandomSampler(train_dataset))
        # datacycler = itertools.cycle(iter(dataloader))

        # for step in range(params['train_steps']):
        #     samples = next(datacycler)

        #     input = tokenize(prompts, model='generator')
            

        #     with torch.no_grad():
        #         generator.eval()
        #         generator_output = generator(input_ids, attention_mask=attention_masks)
        #         responses = generator_output
        #         classifier_input_ids, classifier_attention_masks = tokenize(prompts, responses, model='classifier')

        #         classifier_output = classifier(classifier_input_ids,
        #                                     attention_mask=classifier_attention_masks,
        #                                     labels=labels)
            
        #     generated_labels = classifier_to_generated_labels(classifier_output, labels)
        #     replay_buffer.append(prompts,
        #                         responses,
        #                         labels,
        #                         generated_labels)

    # train loop



    # for epoch in range(1, params['epochs'] + 1):
    #     print(f'Starting Epoch {epoch}')
    #     total_train_loss = 0

    #     for step, (input_ids, attention_masks, labels) in enumerate(dataloader):
    #         input_ids = input_ids.to(device)
    #         attention_masks = attention_masks.to(device)
    #         labels = labels.to(device)
            
    #         outputs = model(input_ids,  
    #                         attention_mask=attention_masks, 
    #                         labels=labels)
    #         loss = outputs.loss

    #         optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #         optimizer.step()

    #         loss = loss.item()
    #         wandb.log({'loss': loss})
    #         total_train_loss += loss

    #         if step % 50 == 0:
    #             prev_avg_eval_loss = sum(running_eval) / len(running_eval)
    #             print(f'Epoch: {epoch}\tStep: {step} \tLoss: {loss:.3f}')
    #             eval_metrics = eval(model, test_dataset, eval_samples=300, **params)
    #             print(eval_metrics)
    #             running_eval.append(eval_metrics['Eval Loss'])
    #             avg_eval_loss = sum(running_eval) / len(running_eval)
    #             if avg_eval_loss > prev_avg_eval_loss:
    #                 stop = 'Eval Loss Increasing, Stopping Early'

    #         if stop:
    #             print(stop)
    #             eval_metrics = eval(model, test_dataset, **params)
    #             print(eval_metrics)
    #             return model

    #     avg_train_loss = total_train_loss / len(dataloader)
    #     print(f'Epoch {epoch}\t\tAvg Training Loss: {avg_train_loss:.3f}')
    #     eval_metrics = eval(model, test_dataset, **params)
    #     print(eval_metrics)
    # return model    
    

# def classifier_to_generated_labels(classifier_output, labels):
#     logits = classifier_output.logits.cpu().numpy()
#     labels = labels.cpu().numpy().flatten()
#     sample_labels = logits[labels]
#     return sample_labels

