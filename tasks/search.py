import os
from typing import List
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from utils import *
from tasks.task import Task

class ConjunctiveSearch(Task):
	
	def __init__(self, 
			     n_objects, 
				 n_trials, 
				 size, 
				 color_names,
				 font_path,
				 **kwargs):
		self.n_objects = n_objects
		self.n_trials = n_trials
		self.size = size
		self.color_names = color_names
		self.font_path = font_path
		super().__init__(**kwargs)
	
	def generate_full_dataset(self):
		img_path = os.path.join(self.data_dir, self.task_name, 'images')
		os.makedirs(img_path, exist_ok=True)
		rgb_values = np.array([mcolors.to_rgb(color) for color in self.color_names])
		img1 = self.letter_img('L')
		img2 = self.letter_img('T')
		metadata_df = pd.DataFrame(columns=['path', 'popout', 'n_objects', 'response', 'answer'])
		for n in tqdm(self.n_objects):
			for i in range(self.n_trials):
				congruent_trial = self.make_trial(img1, img2, rgb_values[0], rgb_values[1], n, False, self.size)
				incongruent_trial = self.make_trial(img1, img2, rgb_values[0], rgb_values[1], n, True, self.size)
				congruent_path = os.path.join(img_path, f'congruent-{n}_{i}.png').split(self.root_dir+'/')[1]
				incongruent_path = os.path.join(img_path, f'incongruent-{n}_{i}.png').split(self.root_dir+'/')[1]
				congruent_trial.save(congruent_path)
				incongruent_trial.save(incongruent_path)
				metadata_df = metadata_df._append({
					'path': congruent_path,
					'incongruent': False,
					'n_objects': n }, ignore_index=True)
				metadata_df = metadata_df._append({
					'path': incongruent_path,
					'incongruent': True,
					'n_objects': n }, ignore_index=True)
		return metadata_df

	
	def make_trial(self,
				   shape1: np.ndarray, 
				   shape2: np.ndarray, 
				   rgb1: np.ndarray, 
				   rgb2: np.ndarray, 
				   n_objects: int = 10, 
				   oddball: bool = True, 
				   img_size: int = 28) -> Image:
		objects = [(shape1, rgb1), (shape2, rgb2)]
		# Add the oddball object first.
		if oddball:
			all_shapes = [shape1]
			all_colors = [rgb2]
			n_objects -= 1
		else:
			all_shapes = []
			all_colors = []
		for _ in range(n_objects):
			random_index = np.random.choice(len(objects))
			all_shapes.append(objects[random_index][0])
			all_colors.append(objects[random_index][1])
		# recolor and resize the shapes
		colored_imgs = [color_shape(img.astype(np.float32), rgb) for img, rgb in zip(all_shapes, all_colors)]
		resized_imgs = np.stack([resize(img, img_size=img_size) for img in colored_imgs])
		np.random.shuffle(resized_imgs) # shuffle the order of the images list
		counting_trial = place_shapes(resized_imgs, img_size=img_size+5) # make shapes a little further apart
		return counting_trial

	def letter_img(self, letter=None):
			assert len(letter)==1 # make sure the string is just a letter.
			img = Image.new('RGB', (32, 32), (255, 255, 255))
			# TODO: CHANGE FONT LOCATION
			font = ImageFont.truetype(self.font_path, 28) 
			draw = ImageDraw.Draw(img)
			draw.text((7, -4), letter, (0,0,0), font=font)
			img_array = np.transpose(np.array(img), (2, 0, 1))
			return img_array


class DisjunctiveSearch(Task):
	
	def __init__(self, 
			     n_objects, 
				 n_trials, 
				 size, 
				 color_names, 
				 shape_inds,
				 **kwargs):
		self.n_objects = n_objects
		self.n_trials = n_trials
		self.size = size
		self.color_names = color_names
		self.shape_inds = shape_inds
		super().__init__(**kwargs)

	def generate_full_dataset(self):
		img_path = os.path.join(self.data_dir, self.task_name, 'images')
		os.makedirs(img_path, exist_ok=True)
		imgs = np.load(os.path.join(self.root_dir, 'imgs.npy'))
		rgb_values = np.array([mcolors.to_rgb(color) for color in self.color_names])
		shape_img = imgs[self.shape_inds[0]]
		metadata_df = pd.DataFrame(columns = ['path', 'popout', 'n_objects', 'response', 'answer'])
		for n in tqdm(self.n_objects):
			for i in range(self.n_trials):
				popout_trial = self.make_trial(shape_img, rgb_values[0], rgb_values[1], n, self.size)
				uniform_trial = self.make_trial(shape_img, rgb_values[0], None, n, self.size)
				uniform_path = os.path.join(img_path, f'uniform-{n}_{i}.png').split(self.root_dir+'/')[1]
				popout_path = os.path.join(img_path, f'popout-{n}_{i}.png').split(self.root_dir+'/')[1]
				uniform_trial.save(uniform_path)
				popout_trial.save(popout_path)
				metadata_df = metadata_df._append({
					'path': uniform_path,
					'popout': False,
					'n_objects': n}, ignore_index=True)
				metadata_df = metadata_df._append({
					'path': popout_path,
					'popout': True,
					'n_objects': n},ignore_index=True)
		return metadata_df

	def make_trial(self, shape: np.ndarray, rgb1: np.ndarray, rgb2: np.ndarray, n_objects: int = 10, img_size: int = 28) -> Image:
		# sample the shapes and colors of objects to include in the trial.
		shape_imgs = shape[np.newaxis].repeat(n_objects, axis=0)
		all_colors = rgb1.reshape(1, -1).repeat(n_objects, axis=0)
		if rgb2 is not None:
			all_colors[0] = rgb2
		# recolor and resize the shapes
		colored_imgs = [color_shape(img.astype(np.float32), rgb) for img, rgb in zip(shape_imgs, all_colors)]
		resized_imgs = [resize(img, img_size=img_size) for img in colored_imgs]
		counting_trial = place_shapes(resized_imgs, img_size=img_size+5)
		return counting_trial