import os
from typing import List
from pathlib import Path
import pandas as pd
import torch

class Task:
    def __init__(self, 
                 task_name=None, 
                 task_variant=None, 
                 model_name=None, 
                 root_dir=None, 
                 output_dir=None, 
                 data_dir=None,
                 metadata_file=None,
                 prompt_path = None):
        self.task_name = task_name
        self.task_variant = task_variant
        self.model_name = model_name
        self.run_id = self.model_name + '_' + self.task_name + '_' + self.task_variant
        self.task_id = self.task_name + '_' + self.task_variant
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.prompt = Path(prompt_path).read_text()
        outpath = os.path.join(self.output_dir, self.task_name)
        self.results_path = os.path.join(outpath, f'{self.model_name}.csv')
        os.makedirs(outpath, exist_ok=True)
        task_path = os.path.join(self.data_dir, self.task_name, self.metadata_file)
        if os.path.exists(self.results_path):
            print(f'Loading task metadata from {self.results_path}...')
            self.results_df = pd.read_csv(self.results_path)
            self.dataset_tensor = torch.load(os.path.join(self.data_dir, self.task_name, 'images.pt'))
        elif os.path.exists(task_path):
            print(f'Loading task metadata from {task_path}...')
            self.results_df = pd.read_csv(task_path)
            self.dataset_tensor = torch.load(os.path.join(self.data_dir, self.task_name, 'images.pt'))
        else:
            print('Generating full dataset...')
            self.results_df, img_tensor = self.generate_full_dataset()
            self.results_df.to_csv(task_path, index=False)
            self.dataset_tensor = img_tensor
            torch.save(img_tensor, os.path.join(self.data_dir, self.task_name, 'images.pt'))
            return None

    def generate_full_dataset(self):
        raise NotImplementedError

    def num_remaining_trials(self):
        pass