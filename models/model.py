import json
import sys
import os
from functools import reduce
from utils import encode_image, get_header
import time
import numpy as np
import pandas as pd
from tasks.task import Task
from tenacity import retry, wait_exponential, stop_after_attempt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import warnings
from typing import Dict
warnings.filterwarnings('ignore')


class Model():

    def __init__(
            self,
            task: Task
    ):
        self.task = task

    def run(self):
        print('Need to specify a particular model class.')
        raise NotImplementedError
    
    def save_results(self, results_file: str=None):
        if results_file:
            self.task.results_df.to_csv(results_file, index=False)
        else:
            filename = f'results_{time.time()}.csv'
            self.task.results_df.to_csv(filename, index=False)

class APIModel(Model):

    def __init__(
            self,
            task: Task,
            model_name: str,
            payload_path: str,
            api_file: str,
            sleep: int = 0,
            shuffle: bool = False,
            n_trials: int = None,
    ):
        self.model_name = model_name
        good_models = ['gpt4v', 'gpt4o', 'claude-sonnet', 'claude-opus', 'gemini-ultra', 'dalle', 'stable-diffusion', 'imagen']
        assert self.model_name in good_models, f'Model name must be one of {str(good_models)}, not {self.model_name}'
        if task:
            super().__init__(task)
            self.results_file = self.task.results_path
        self.payload = json.load(open(payload_path))
        self.api_metadata = json.load(open(api_file, 'r'))
        self.header = get_header(self.api_metadata, model=self.model_name)
        self.api_metadata = self.api_metadata[self.model_name]
        self.sleep = sleep
        self.shuffle = shuffle
        self.n_trials = n_trials
        # shuffle and subsample the dataset, if necessary
        if self.shuffle and task:
            if self.n_trials:
                self.task.results_df = self.task.results_df.sample(n=self.n_trials)
            else:
                self.task.results_df = self.task.results_df.sample(frac=1)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(10))
    def run(self):
        p_bar = tqdm(total=self.task.num_remaining_trials())
        for i, trial in self.task.results_df.iterrows():
            if type(trial.response) != str or trial.response == '0.0':
                trial_payload = self.payload.copy()
                task_payload = self.build_vlm_payload(trial, trial_payload)
                response = self.run_trial(self.header, self.api_metadata, task_payload)
                self.task.results_df.loc[i, 'response'] = response
                p_bar.update()
                time.sleep(self.sleep)
            if i % 50 == 0:
                self.save_results(self.results_file)
        self.save_results(self.results_file)
        return self.task.results_df


class LocalLanguageModel(Model):

    def __init__(
        self,
        task: Task = None,
        max_parse_tokens: int = 256,
        prompt_format: str = None,
        weights_path: str = None,
        probe_layers: Dict = None
    ):
        super().__init__(task)
        if task:
            self.results_file = self.task.results_path
        self.max_parse_tokens = max_parse_tokens
        self.prompt_format = prompt_format
        self.weights_path = weights_path
        self.probe_layers = probe_layers
        self.prompt = ''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.activations = {}
        self.llm = AutoModelForCausalLM.from_pretrained(self.weights_path, device_map='auto', torch_dtype='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(self.weights_path, use_fast=True)
        self.llm.eval()
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.target_column = 'response'

    def run_batch(self, batch):
        prompts = [p.format(text_to_parse=t) for p, t in zip([self.prompt]*len(batch), batch[self.target_column])]
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding='longest',
            truncation=True,
        )
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=self.max_parse_tokens)
        outputs = [output[inputs.input_ids.shape[1]:] for output in outputs]
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, o in zip(prompts, decoded_outputs):
            print(f'Prompt: {i}\nResponse: {o}\n')
        batch['answer'] = decoded_outputs
        return batch

class LocalVLModel(Model):

    def __init__(
        self,
        task: Task,
        parse_model: LocalLanguageModel = None,
        max_vision_tokens: int = 512,
        max_parse_tokens: int = 256,
        parse_response: bool = True,
        batch_size: int = 32,
        prompt_format: str = None,
        weights_path: str = None,
        probe_layers: Dict = None
    ):
        super().__init__(task)
        self.results_file = self.task.results_path
        self.parse_response = parse_response
        self.parse_model = parse_model
        self.max_vision_tokens = max_vision_tokens
        self.max_parse_tokens = max_parse_tokens
        self.batch_size = batch_size
        self.prompt_format = prompt_format
        self.weights_path = weights_path
        self.probe_layers = probe_layers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.prompt = self.prompt_format.format(question=self.task.prompt)
        if self.parse_response and self.parse_model:
            self.parse_model.prompt = self.parse_model.prompt_format.format(question=self.task.parse_prompt + '\n{text_to_parse}')
        self.activations = {}


    def run(self):
        results = []
        batches = np.array_split(self.task.results_df, np.ceil(len(self.task.results_df)/self.batch_size))
        for i, batch in enumerate(batches):
            batch = self.run_batch(batch, i)
            results.append(batch)

            # TODO: CHANGE TO SAVE FULL DATAFRAME EVERY 50 STEPS
            if i % 10 == 0:
                self.task.results_df = pd.concat(batches[i:]+results)
                self.save_results(self.results_file)
        self.save_activations()

    def getActivations(self, name):
        def hook(model, input, output):
            try:
                print(f'{name} Shape: {output.shape}')
                self.activations[name].append(output.detach().cpu().numpy())
            except KeyError:
                self.activations[name] = [output.detach().cpu().numpy()]
        return hook
    
    def register_hooks(self):
        for layer, names_list in self.probe_layers.items():
            print(f'Registering hook for {layer}')
            target = reduce(getattr, names_list, self.model)
            print(f'Target: {target}')
            _ = target.register_forward_hook(self.getActivations(layer))

    def save_activations(self):
        outpath = os.path.join(self.task.output_dir, self.task.task_id, self.task.model_name)
        os.makedirs(outpath, exist_ok=True)
        for layer, layer_activations in self.activations.items():
            try:
                activations = np.stack([act.reshape(self.batch_size, -1) for act in layer_activations])
                np.save(os.path.join(outpath, f'{layer}.npy'), activations)
            except:
                print(f'Error saving {layer}')
