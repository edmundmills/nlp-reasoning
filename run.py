import argparse
import os
import random

from hydra import compose, initialize
from flatten_dict import flatten
import numpy as np
from omegaconf import OmegaConf
import torch
import transformers
import wandb


from nlp_reasoning.classifier import Classifier
from nlp_reasoning.generator import Generator
from nlp_reasoning.dataset import ClassifierDataset, GeneratorDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()
    return args

def get_config(args):
    with initialize(config_path='./config'):
        cfg = compose('config.yaml', overrides=args.overrides)

    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.classifier.debug = cfg.debug
    cfg.generator.debug = cfg.debug
    cfg.reasoner.debug = cfg.debug
    print(OmegaConf.to_yaml(cfg))        
    return cfg

def flatten_args(args):
    return flatten(OmegaConf.to_container(args, resolve=True), reducer='dot')

class wandb_run:
    def __init__(self, args):
        self.args = args

    def __enter__(self):
        if not self.args.debug:
            wandb.init(
                entity='nlp-reasoning',
                project=self.args.name,
                notes="",
                config=flatten_args(self.args),
            )

    def __exit__(self, exc_type, exc_value, exc_tb):
        if not self.args.debug:
            wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    args = get_config(args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = ClassifierDataset('sarcasm_headlines')
    classifier = Classifier()
    if args.classifier.name == 'pretrain_classifier':
        pretrain_args = args.classifier
        with wandb_run(pretrain_args):
            classifier.fine_tune(dataset, pretrain_args)
    else:
        classifier.load()

    generator = Generator()
    if args.generator.name == 'pretrain_generator':
        generator_dataset = GeneratorDataset('winning_arguments')
        pretrain_args = args.generator
        with wandb_run(pretrain_args):
            generator.fine_tune(generator_dataset, pretrain_args)

    headlines = [dataset[idx]['text'] for idx in range(10)]
    responses = generator.generate(headlines)
    for response in responses:
        print(response)


