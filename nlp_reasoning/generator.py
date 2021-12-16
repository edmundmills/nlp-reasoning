from collections import deque
import itertools
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Adafactor, AutoTokenizer
import wandb

from nlp_reasoning.dataset import ReasoningSample, ReplayBuffer
from nlp_reasoning.data_utils import trim_trailing_sentence, remove_linebreaks
from nlp_reasoning.model import Model

class Tokenizer:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, sample:Dict) -> Dict:
        prompt = sample['prompt']
        response = sample['response']
        output = self.tokenizer.encode_plus(text=prompt,
                                            text_pair=response,
                                            add_special_tokens = True,
                                            max_length = 75,
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
        print(f'Loading Pretrained Generator Model ({self.model_class.__name__})...')
        self.model = self.model_class.from_pretrained("EleutherAI/gpt-neo-1.3B")
        print('Model Loaded. Transfering to GPU...')
        self.model = self.model.to(self.device)
        print('Model transfered to GPU.')
        self.tokenizer = Tokenizer()

    def generate(self, prompts:List[str]) -> List[str]:
        responses = []
        with Model.eval_mode(self.model):
            for prompt in prompts:
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                gen_tokens = self.model.generate(input_ids, do_sample=True, temperature=0.9, min_length=40, max_length=60)
                gen_text = self.tokenizer.batch_decode(gen_tokens.cpu())[0]
                gen_text = trim_trailing_sentence(gen_text)
                gen_text = remove_linebreaks(gen_text)
                responses.append(gen_text)
        return responses

    def fine_tune(self, dataset, args):
        if not args.debug:
            self.model_dir = Path(f'./models/generator/{wandb.run.name}')

        train_data, test_data = dataset.train_test_split(0.8)
        train_data = train_data.to_tokenized_tensors(tokenizer=self.tokenizer)
        test_data = test_data.to_tokenized_tensors(tokenizer=self.tokenizer)
        
        self.train(train_data, test_data, args)

        if not args.debug:
            self.save()


    def train(self, train_dataset, test_dataset, args):
        dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=4,
                                drop_last=True,
                                sampler=RandomSampler(train_dataset))
        
        optimizer = Adafactor(self.model.parameters(),
                              lr=args.lr,
                              scale_parameter=False, relative_step=False, warmup_init=False)
        optimizer.zero_grad()

        self.model.train()

        running_loss = 0

        for epoch in range(1, args.epochs + 1):
            print(f'Starting Epoch {epoch}')
            total_train_loss = 0

            for step, (input_ids, attention_masks) in enumerate(dataloader):
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                
                outputs = self.model(input_ids,  
                                     attention_mask=attention_masks, 
                                     labels=input_ids)
                loss = outputs.loss
                loss.backward()

                loss = loss.item()
                total_train_loss += loss
                running_loss += loss

                if ((step + 1) / args.gradient_accumulation) % 20 == 0:
                    print(f'Epoch: {epoch}\tStep: {step + 1} \tLoss: {running_loss / args.gradient_accumulation:.3f}')

                if (step + 1) % args.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    wandb.log({'loss': running_loss / args.gradient_accumulation})
                    running_loss = 0

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            avg_train_loss = total_train_loss / len(dataloader)
            print(f'Epoch {epoch}\t\tAvg Training Loss: {avg_train_loss:.3f}')

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

