import os
import base64
from glob import glob
from typing import Dict

import numpy as np
from PIL import Image


def paste_shape(shape: np.ndarray, 
                positions: np.ndarray, 
                canvas_img: Image.Image, 
                i: int, 
                img_size: int = 40) -> np.ndarray:
    '''
    Paste a shape onto a canvas image at a random position.

    Parameters:
    shape (np.ndarray): The shape to be pasted.
    positions (np.ndarray): The positions of the shapes on the canvas.
    canvas_img (Image.Image): The canvas image.
    i (int): The index of the current shape.
    img_size (int): The size of the shape. Default is 12.

    Returns:
    np.ndarray: The updated positions of the shapes on the canvas.
    '''
    img = Image.fromarray(np.transpose(shape, (1, 2, 0)))
    position = np.array(np.random.randint(0, 256-img_size, size=2)).reshape(1,-1)
    # Keep trying to find a position that is far enough from the other shapes.
    while np.any(np.linalg.norm(positions-position, axis=1) < img_size):
        position = np.array(np.random.randint(0, 256-img_size, size=2)).reshape(1,-1)
    canvas_img.paste(img, tuple(position.squeeze()))
    positions[i] = position
    return positions


def color_shape(img: np.ndarray, rgb: np.ndarray, bg_color: float = 1, all_black: bool = False) -> np.ndarray:
    '''
    Color a grayscale image with a given RGB code.

    Parameters:
    img (np.ndarray): The grayscale image.
    rgb (np.ndarray): The RGB code.
    bg_color (float): The background color. Default is 1.
    all_black (bool): Whether to color the image black. Default is False.

    Returns:
    np.ndarray: The colored image.
    '''
    if all_black:
        rgb = np.ones(3)
        return img.astype(np.uint8) * rgb.reshape((3,1,1))
    # Normalize the RGB code.
    rgb = rgb.astype(np.float32)
    if rgb.max() > 1:
        rgb /= rgb.max()  # normalize rgb code
    img /= img.max()  # normalize image
    colored_img = (1-img) * rgb.reshape((3,1,1))
    colored_img += img * bg_color
    return (colored_img * 255).astype(np.uint8)


def resize(image: np.ndarray, img_size: int=28) -> np.ndarray:
    '''
    Resize an image to a given size.

    Parameters:
    image (np.ndarray): The image to be resized.
    size (int): The size to resize the image to. Default is 12.

    Returns:
    np.ndarray: The resized image.
    '''
    image_array = np.transpose(image, (1, 2, 0))
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    resized_image = image.resize((img_size, img_size), Image.LANCZOS)
    return np.transpose(np.array(resized_image), (2, 0, 1))


def encode_image(image_path):
    '''
    Encode an image as a base64 string.

    Parameters:
    image_path (str): The path to the image.
    '''
    # check if the image path exists
    if not os.path.exists(image_path):
        # find the rest of the path after the folder named `data` in a sub-folder named `vlm-binding`
        image_path = os.path.join('vlm-binding', *image_path.split('data')[1:])
        if image_path.startswith('/'):
            image_path = image_path[1:]
        image_path = os.path.join(os.path.dirname(__file__), 'data', image_path)

    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def place_shapes(shape_imgs, img_size=32):
    # Define the canvas to draw images on, font, and drawing tool.
    canvas = np.ones((3, 256, 256), dtype=np.uint8) * 255
    canvas = np.transpose(canvas, (1, 2, 0))  # Transpose to (256x256x3) for PIL compatibility.
    canvas_img = Image.fromarray(canvas)
    # Add the shapes to the canvas.
    n_shapes = len(shape_imgs)
    positions = np.zeros([n_shapes, 2])
    for i, img in enumerate(shape_imgs):
        positions = paste_shape(img, positions, canvas_img, i, img_size=img_size)
    return canvas_img


def get_header(api_info, model=None) -> Dict[str, str]:
    api_key = api_info[model]['api_key']
    if model == 'gpt4o' or model == 'gpt4v':
        return {
            'Content-Type': 'application/json',
            'api-key': api_key
        }
    elif model == 'claude-sonnet' or model == 'claude-opus':
        return {
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
            'x-api-key': api_key
        }
    elif model == 'gemini-ultra':
        return {
            'Content-Type': 'application/json',
            'x-goog-api-key': api_key
        }
    elif model == 'dalle':
        return {
            'Content-Type': 'application/json',
            'api-key': api_key
        }
    elif model == 'stable-diffusion':
        return {
            'authorization': f'Bearer {api_key}',
            'accept': 'application/json'
        }
    elif model == 'parti':
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer'
            }
    elif model == 'openai':
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
    else: 
        raise ValueError(f'Model {model} not recognized.')