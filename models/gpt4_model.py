import ast
import requests
from utils import encode_image
from models.model import APIModel
from tenacity import retry, wait_exponential, stop_after_attempt
import base64
import httpx
import os


class GPT4Model(APIModel):

    def __init__(self, max_tokens: int = 512, **kwargs):
        self.max_tokens = max_tokens
        super().__init__(**kwargs)
        self.payload['max_tokens'] = self.max_tokens

    def build_vlm_payload(self, trial_metadata, task_payload):
        '''
        Parameters:
            trial_metadata (dict): The metadata for the task.
            task_payload (str): The task payload.

        Returns:
            str: The parsed task prompt.
        '''
        task_payload['messages'][0]['content'][0]['text'] = self.task.prompt
        task_payload['messages'][0]['content'] = [task_payload['messages'][0]['content'][0]]
        img_path = trial_metadata['path']
        images = [encode_image(img_path)]

        # Add the images to the payload
        image_payload = [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image}'}} for image in images]
        task_payload['messages'][0]['content'] += image_payload
        return task_payload

    def run_trial(self, header, api_metadata, task_payload):
        '''
        Parameters:
            header (dict): The header for the API request.
            api_metadata (dict): The metadata for the API.
            task_payload (dict): The payload for the task.

        Returns:
            str: The response.
        '''
        
        # Until the model provides a valid response, keep trying.
        trial_response = requests.post(
            api_metadata['endpoint'],
            headers=header,
            json=task_payload,
            timeout=240
        )

        # Check for easily-avoidable errors.
        if 'error' in trial_response.json():
            raise ValueError('Returned error: \n' + str(trial_response.json()['error']['message']))
        response = trial_response.json()['choices'][0]['message']['content']
        return response
    

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(10))
    def run_single_request(self, prompt: str, image: str):
        if os.path.exists(image):
            with open(image, 'rb') as image_file:
                image = base64.b64encode(image_file.read()).decode('utf-8')
            image_payload = [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image}'}}]
        # Otherwise, try to load the image URL.
        else:
            try:
                image = base64.b64encode(httpx.get(image).content).decode('utf-8')
                media_type = 'image/jpeg'
                image_payload = [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image}'}}]
            
            except requests.exceptions.MissingSchema as e:
                raise ValueError('Image URL not found.', f'Error: {e}')
        # Construct the payload.
        payload = self.payload.copy()
        payload['messages'][0]['content'][0]['text'] = prompt
        payload['messages'][0]['content'] = [payload['messages'][0]['content'][0]]
        payload['messages'][0]['content'] += image_payload
        response = self.run_trial(self.header, self.api_metadata, payload)
        return response