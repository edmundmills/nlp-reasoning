import os
from pathlib import Path

import torch

class Model:
    def __init__(self) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def save(self):
        print(f'Saving model to {self.model_dir}')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.model_dir)
        print(f'Model Saved.')

    def load(self, model_dir=None):
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            models = Path(f'models/{self.__class__.__name__.lower()}')
            model_dirs = [dir for dir in models.iterdir() if dir.is_dir()
                          and dir / 'config.json' in list(dir.iterdir())]
            model_dirs = sorted(model_dirs, key=lambda t: -os.stat(t).st_mtime)
            self.model_dir = model_dirs[0]
        print(f'Loading {self.model_class.__name__} as classifier from {str(self.model_dir)}')
        self.model = self.model_class.from_pretrained(self.model_dir)
        self.model.to(self.device)

    class eval_mode:
        def __init__(self, model) -> None:
            self.model = model

        def __enter__(self):
            self.model.eval()

        def __exit__(self, exc_type, exc_value, exc_tb):
            self.model.train()