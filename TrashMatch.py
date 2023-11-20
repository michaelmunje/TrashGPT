# This script will take input from TrashGPT and TrashOD
# It will combine their results, with TrashGPT containing oracle labels but relative locations
# And TrashOD containing relative locations but noisy labels

import numpy as np
from copy import copy

class TrashMatch:
    def __init__(self):
        pass
        
    def euclidean_distance(self, location_a, location_b) -> float:
        return np.sqrt((location_a[0] - location_b[0])**2 + (location_a[1] - location_b[1])**2)
    
    def get_yx(self, y1, x1, y2, x2) -> [float, float]:
        return [y1 + ((y2 - y1) / 2), x1 + ((x2 - x1) / 2)]
    
    def match_results(self, gpt_result, od_results: dict, frame_height: int, frame_width: int):
        """
        Matches the results from GPT and OD.
        """
        
        if len(gpt_result.name) is None:
            return gpt_result
        
        # let's convert od results to relative center coordinates
        od_results_yx = []
        for i in range(len(od_results['detection_boxes'])):
            bb = od_results['detection_boxes'][i]
            od_results_yx.append(self.get_yx(bb[0], bb[1], bb[2], bb[3]))
            assert od_results_yx[-1][0] >= 0 and od_results_yx[-1][0] <= 1
            assert od_results_yx[-1][1] >= 0 and od_results_yx[-1][1] <= 1

        # let's convert gpt results to relative center coordinates
        matched_result = copy(gpt_result)
        assert type(matched_result.location[0]) == int
        assert type(matched_result.location[1]) == int
        matched_result.location[0] /= frame_height
        matched_result.location[1] /= frame_width

        # Now, let's find the closest match for each item in gpt_results
        distance_threshold = 0.4
        # find the closest match
        closest_match_distance = 1000000
        old_location = gpt_result.location
        for i, loc in enumerate(od_results_yx):
            if od_results['detection_scores'][i] < 0.2:
                continue
            distance = self.euclidean_distance(old_location, loc)
            if distance < closest_match_distance and distance < distance_threshold:
                closest_match_distance = distance
                matched_result.location = loc

        # convert back to pixel coordinates
        matched_result.location[0] = int(matched_result.location[0] * frame_height)
        matched_result.location[1] = int(matched_result.location[1] * frame_width)

        return matched_result