import json
from pathlib import Path
import re

from convokit import download

def download_winning_args():
    parent_dir = Path('data_test2')
    parent_dir.mkdir(exist_ok=True)
    dataset_name = 'winning-args-corpus'
    path = download(dataset_name, data_dir=parent_dir)
    return path

def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

def clean_text(text):
    if text:
        return re.sub(r"(@\[A-Za-z0-9]+)|(\w+:\/\/\S+)|http.+?", "", text)
    else:
        return text

def trim_trailing_sentence(text):
    text = '.'.join(text.split('.')[:-1])
    if text:
        text += '.'
    return text