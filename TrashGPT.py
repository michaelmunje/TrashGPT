from openai import OpenAI
import requests
import os


class Trash:
    """
    Represents a detected trash object in an image.
    
    Attributes:
        name (str): The name of the trash object.
        location (list of int): The location of the object in the image, given as [y, x] coordinates.
    """
    def __init__(self, name: str, location: [int, int]):
        self.name = name
        self.location = location # [y, x]


class TrashGPT:
    def __init__(self):
        self.client = OpenAI()

        self.trash_prompt = "Is there any trash in this image?" + \
            " Provide the response as a brief list separated by semi-colons" + \
            " and nothing else, where the list contains 'Object Name: Relative Coordinates (Y, X)'." + \
            "\nExample: 'Soda can: 0.5, 0.5; Bottle: 0.2, 0.8'."

        api_key = os.environ.get('OPENAI_API_KEY')
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        del api_key
        self.model_name = "gpt-4-vision-preview"
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def get_payload(self, image, prompt=None):
        if prompt is None:
            prompt = self.trash_prompt

        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}"
                                }
                            }
                        ]
                    }
                ],
            "max_tokens": 300
        }

    def perform_trash_detection(self, image, image_height, image_width) -> [Trash]:
        payload = self.get_payload(image)
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        output = response.json()
        gpt_output =  output['choices'][0]['message']['content']
        pieces_of_trash = gpt_output.split(';')
        try:
            trash = []
            for item in pieces_of_trash:
                contents = item.split(':')
                trash_name = contents[0]
                trash_locations = contents[-1].split(',')
                trash_y = int(float(trash_locations[0]) * image_height)
                trash_x = int(float(trash_locations[1]) * image_width)
                trash.append(Trash(trash_name, [trash_y, trash_x]))
        except:
            trash = []
        return trash
    
    def overlay_results(self, image, trash_items: [Trash]):
        """
        Overlays red dots on the image at the given locations.
        """
        image = image.copy()
        for item in trash_items:
            location = item.location
            image[location[0] - 3:location[0] + 3, location[1] - 3:location[1] + 3] = [0, 0, 255]
        return image
