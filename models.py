import os
from pathlib import Path

from transformers import BertForSequenceClassification

def load_classifier():
    classifiers = Path('models/classifier')
    classifier_dirs = [dir for dir in classifiers.iterdir() if dir.is_dir()]
    classifier_dirs = sorted(classifier_dirs, key=lambda t: -os.stat(t).st_mtime)
    classifier_dir = classifier_dirs[0]
    print(f'Loading classifier from {str(classifier_dir)}')
    classifier = BertForSequenceClassification.from_pretrained(classifier_dir)
    return classifier
