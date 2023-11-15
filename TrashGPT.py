from openai import OpenAI
import base64
import requests
import numpy as np
import cv2
import time
import os
import TrashObjectDetector


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
    
    def execute_webcam_demo(self):
        try:
            cap = cv2.VideoCapture(0)
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()

                if not ret:
                    print("Failed to grab frame")
                    break

                # Display the captured frame
                cv2.imshow('Webcam', frame)

                # Break the loop with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # Save the captured frame
                    timestamp = int(time.time())
                    filename = f'image_{timestamp}.jpg'
                    cv2.imwrite(filename, frame)

                    trash_detector_od = TrashObjectDetector.TrashObjectDetector()

                    result = trash_detector_od.detect(frame)
                    frame_copy = frame.copy()
                    od_img = trash_detector_od.overlay_bounding_boxes(frame_copy, result)
                    # save od_img
                    new_filename = filename.split('.')[0] + '_OD.jpg'
                    cv2.imwrite(new_filename, od_img)

                    trash_result = trash_detector_od.filter_for_trash_items(result)
                    frame_copy = frame.copy()
                    od_trash_img = trash_detector_od.overlay_bounding_boxes(frame_copy, trash_result)
                    # save od_img
                    new_filename = filename.split('.')[0] + '_OD_Trash.jpg'
                    cv2.imwrite(new_filename, od_trash_img)

                    # Function to encode the image
                    with open(filename, "rb") as image_file:
                        image = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    trash = self.perform_trash_detection(image, frame.shape[0], frame.shape[1])
                    
                    # add another part below the image that has a white background
                    # make an img that just white
                    # get width of img with height of 200
                    width = frame.shape[1]
                    
                    lines = ['GPT-4V:']
                    
                    if len(trash) > 0:
                        for item in trash:
                            # convert item.location to relative coordinates
                            x = round(item.location[1] / width, 2)
                            y = round(item.location[0] / frame.shape[0], 2)
                            lines.append(f'{item.name}: {y}, {x}')
                    else:
                        lines.append('No trash detected.')

                    caption_img = np.ones((200, width, 3), np.uint8) * 255
                    
                    for i in range(len(lines)):
                        caption_img = cv2.putText(img = caption_img,
                                                    text = lines[i], 
                                                    org = (10, 35 + 35 * i), 
                                                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                                                    fontScale = 1,
                                                    color = (0, 0, 0), 
                                                    thickness = 2, 
                                                    lineType = cv2.LINE_AA)
                    # Add images together
                    frame = np.concatenate((frame, caption_img), axis=0)

                    # same thing but above
                    prompt_img = np.ones((250, width, 3), np.uint8) * 255
                    
                    prompt_and_example = self.trash_prompt.split('\n')
                    prompt = prompt_and_example[0]
                    example = prompt_and_example[-1]

                    # break up prompt into lines when characters > 75
                    lines = ['Human:']
                    line = ''
                    for word in prompt.split(' '):
                        if len(line) + len(word) > 75:
                            lines.append(line)
                            line = ''
                        line += word + ' '
                    lines += [line]
                    lines += [example]
                    
                    for i in range(len(lines)):
                        # place text on img
                        prompt_img = cv2.putText(img= prompt_img, 
                                                 text = lines[i],
                                                 org = (10, 35 + 50 * i), 
                                                 fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                                                 fontScale = 0.5, 
                                                 color = (0, 0, 0), 
                                                 thickness = 1, 
                                                 lineType = cv2.LINE_AA)

                    frame = np.concatenate((prompt_img, frame), axis=0)

                    # save img
                    new_filename = filename.split('.')[0] + '_GPT.jpg'
                    cv2.imwrite(new_filename, frame)
                    print('Saved image to', filename)
                    break
        finally:
            # When everything is done, release the capture
            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    gpt = TrashGPT()
    gpt.execute_webcam_demo()