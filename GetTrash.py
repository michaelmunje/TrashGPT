import cv2
import time
import numpy as np
import TrashObjectDetector
import TrashGPT
import TrashMatch
import base64


class GetTrash:
    def __init__(self):        
        self.trash_detector_gpt = TrashGPT.TrashGPT()
        self.trash_detector_od = TrashObjectDetector.TrashObjectDetector()
        self.trash_match = TrashMatch.TrashMatch()
        
    def get_trash(self, image):
        # get original dimensions
        result = self.trash_detector_od.detect(image)
        trash_result = self.trash_detector_od.filter_for_trash_items(result)

        # save file
        timestamp = int(time.time())
        filename = f'image_{timestamp}.jpg'
        cv2.imwrite(filename, image)
        # Function to encode the image
        with open(filename, "rb") as image_file:
            gpt_img_input = base64.b64encode(image_file.read()).decode('utf-8')
        
        trash = self.trash_detector_gpt.perform_trash_detection(gpt_img_input, image.shape[0], image.shape[1],
                                                                n_runs=5)
        
        # finally, let's do the matching
        match_results = self.trash_match.match_results(trash, trash_result, image.shape[0], image.shape[1])
        
        location_results = [match_results.location]
        return location_results
    
    def get_trash_can(self, image):
        result = self.trash_detector_od.detect(image)
        trash_result = self.trash_detector_od.filter_for_trash_items(result)

        # save file
        timestamp = int(time.time())
        filename = f'image_{timestamp}.jpg'
        cv2.imwrite(filename, image)
        # Function to encode the image
        with open(filename, "rb") as image_file:
            gpt_img_input = base64.b64encode(image_file.read()).decode('utf-8')
        
        trash = self.trash_detector_gpt.perform_trash_can_detection(gpt_img_input, image.shape[0], image.shape[1])
        
        # finally, let's do the matching
        match_results = self.trash_match.match_results(trash, trash_result, image.shape[0], image.shape[1])
        
        location_results = [p.location for p in match_results]
        return location_results
