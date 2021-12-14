import argparse
import os
import random

import numpy as np
import torch

import wandb

from train_classifier import train_classifier
# from train_generator import train_generator
from models import load_classifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', dest='debug',
                        action='store_true', default=False)
    parser.add_argument('--train-classifier', dest='train_classifier',
                        action='store_true', default=False)
    args = parser.parse_args()
    if args.train_classifier:
        args.classifier_training_batch_size = 32
        args.classifier_training_lr = 2e-5
        args.classifier_training_epochs = 4

    args.seed = 42
    args.train_steps = 1e5
    args.generator_batch_size = 4
    args.generator_lr = 2e-5
    args.classifier_batch_size = 32
    print(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.train_classifier:
        wandb.init(
            entity='nlp-reasoning',
            project='train_classifier',
            notes="",
            config=args,
        )
        classifier = train_classifier(args)
        wandb.finish()
    else:
        classifier = load_classifier()

    # wandb.init(
    #     entity='nlp-reasoning',
    #     project='train_generator',
    #     notes="",
    #     config=args,
    # )

    # generator = train_generator(classifier, args)
    
    # wandb.finish()