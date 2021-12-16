import argparse
import os
import random

from hydra import compose, initialize
from flatten_dict import flatten
import numpy as np
from omegaconf import OmegaConf
import torch
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
        if args.wandb:
            wandb.init(
                entity='nlp-reasoning',
                project=args.project_name,
                notes="",
                config=flatten_args(self.args),
            )

    def __exit__(self, exc_type, exc_value, exc_tb):
        if args.wandb:
            wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    args = get_config(args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = ClassifierDataset('sarcasm_headlines')
    classifier = Classifier()
    if args.classifier == 'pretrain':
        pretrain_args = args.classifier
        with wandb_run(pretrain_args):
            classifier.fine_tune(dataset, pretrain_args)
    else:
        classifier.load()

    generator = Generator()
    headlines = [dataset[idx]['text'] for idx in range(10)]
    responses = generator.generate(headlines)
    for response in responses:
        print(response)


