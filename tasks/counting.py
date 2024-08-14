import os
from typing import List
from glasbey import create_palette
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *
from tasks.task import Task
from pathlib import Path


class Counting(Task):
    def __init__(self,
                n_objects: List[int],
                n_trials: int,
                min_size: int,
                max_size: int,
                **kwargs):
        self.n_objects = n_objects
        self.n_trials = n_trials
        self.min_size = min_size
        self.max_size = max_size
        super().__init__(**kwargs)

    def generate_full_dataset(self):
        img_path = os.path.join(self.data_dir, self.task_name, 'images')
        os.makedirs(img_path, exist_ok=True)
        imgs = np.load(os.path.join(self.root_dir, 'imgs.npy'))
        metadata_df = pd.DataFrame(columns=['path', 'n_objects', 'response'])
        palette = create_palette(palette_size=max(self.n_objects), grid_size=256, grid_space='JCh')
        rgb_colors = np.array([mcolors.hex2color(color) for color in palette])
        for n in tqdm(self.n_objects):
            for i in range(self.n_trials):
                shape_inds = np.random.choice(len(imgs), n, replace=False)
                color_inds = np.random.choice(len(rgb_colors), n, replace=False)
                colors = rgb_colors[color_inds]
                shapes = imgs[shape_inds]
                trial = self.make_trial(shapes, colors, size_range=(self.min_size, self.max_size))
                trial_path = os.path.join(img_path, f'nObjects={n}_trial={i}.png').split(self.root_dir+'/')[1]
                trial.save(trial_path)
                metadata_df = metadata_df._append({'path': trial_path, 'n_objects': n }, ignore_index=True)
        return metadata_df

    def make_trial(self, shape_imgs, colors, size_range=(10, 20)):
        sizes = np.random.randint(size_range[0], size_range[1], len(shape_imgs))
        colored_imgs = [color_shape(img.astype(np.float32), rgb) for img, rgb in zip(shape_imgs, colors)]
        resized_imgs = [resize(img, img_size=size) for img, size in zip(colored_imgs, sizes)]
        counting_trial = place_shapes(resized_imgs, img_size=max(sizes))
        return counting_trial