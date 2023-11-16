# Parts of this code were adapted from the TensorFlow Object Detection API:
# https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py

import pathlib
import tensorflow as tf
import numpy as np
import urllib
import collections
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import re
import os


class TrashObjectDetector:
    def __init__(self):
        self.trash_labels = [
            'bottle', 'can', 'cup', 'bag', 'straw', 'wrapper', 'container', 
            'material', 'paper', 'cardboard', 'battery', 'phone', 'electronics', 
            'bulb', 'boards', 'syringe', 'medication', 'chemical', 'paint', 'pesticide', 
            'cigarette', 'mask', 'gloves', 'toys', 'clothes', 'shoes', 'tire', 
            'glass', 'balloon', 'scrap', 'peel', 'waste', 'newspaper', 'magazine']

        self.STANDARD_COLORS = [
            'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
            'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
            'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
            'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
            'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
            'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
            'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
            'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
            'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
            'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
            'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
            'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
            'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
            'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
            'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
            'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
            'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
            'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
            'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
            'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
            'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
            'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
            'WhiteSmoke', 'Yellow', 'YellowGreen'
            ]
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # Adjust the model name as needed
        model_name = 'tf2/20200711/efficientdet_d4_coco17_tpu-32'
        self.load_model(model_name)

        # Load label map for COCO dataset
        self.load_label_map()
        
    def load_label_map(self):
        url = 'https://gist.githubusercontent.com/iitzco/3b2ee634a12f154be6e840308abfcab5/raw/89d26e0aa56b0c356fb78cc2408fc64f4197b88a/coco_label_map.py'
        response = urllib.request.urlopen(url)
        label_map_raw = response.read().strip()
        label_map_raw = label_map_raw.replace(b'\n', b'')
        label_map_raw = label_map_raw.replace(b'LABEL_MAP = ', b'')
        label_map_raw = label_map_raw.replace(b"\'", b"\"")
        # search through byte string, if it is an integer, enclose in double quotes
        # (json only accepts double quotes)
        label_map_raw = re.sub(b'([0-9]+)', b'"\g<1>"', label_map_raw)
        map_with_str_keys = json.loads(label_map_raw.decode('utf-8'))
        # convert keys back to ints
        self.label_map = {}
        for key in map_with_str_keys:
            self.label_map[int(key)] = map_with_str_keys[key]
        del map_with_str_keys
    
    # Load a pre-trained model from TensorFlow model zoo
    def load_model(self, model_name):
        base_url = 'http://download.tensorflow.org/models/object_detection/'
        model_file = model_name + '.tar.gz'
        model_dir = tf.keras.utils.get_file(
            fname=None,
            origin=base_url + model_file,
            untar=True)

        model_dir = pathlib.Path(model_dir)/"saved_model"
        self.model = tf.saved_model.load(str(model_dir))
        

    def draw_bounding_box_on_image(self,
                                   image,
                                   ymin,
                                   xmin,
                                   ymax,
                                   xmax,
                                   color='red',
                                   thickness=4,
                                   display_str_list=()):
        """Adds a bounding box to an image.

        Bounding box coordinates can be specified in either absolute (pixel) or
        normalized coordinates by setting the use_normalized_coordinates argument.

        Each string in display_str_list is displayed on a separate line above the
        bounding box in black text on a rectangle filled with the input 'color'.
        If the top of the bounding box extends to the edge of the image, the strings
        are displayed below the bounding box.

        Args:
            image: a PIL.Image object.
            ymin: ymin of bounding box.
            xmin: xmin of bounding box.
            ymax: ymax of bounding box.
            xmax: xmax of bounding box.
            color: color to draw bounding box. Default is red.
            thickness: line thickness. Default value is 4.
            display_str_list: list of strings to display in box
                            (each to be shown on its own line).
            use_normalized_coordinates: If True (default), treat coordinates
            ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
            coordinates as absolute.
        """
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
        if thickness > 0:
            draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                    (left, top)],
                    width=thickness,
                    fill=color)
        try:
            font = ImageFont.truetype('arial.ttf', 24)
        except IOError:
            font = ImageFont.load_default()

        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        display_str_heights = [font.getbbox(ds)[3] for ds in display_str_list]
        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height
        # Reverse list and print from bottom to top.
        for display_str in display_str_list[::-1]:
            bbox = font.getbbox(display_str)
            text_width, text_height = bbox[2], bbox[3]
            margin = np.ceil(0.05 * text_height)
            draw.rectangle(
                [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                text_bottom)],
                fill=color)
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill='black',
                font=font)
            text_bottom -= text_height - 2 * margin
    
    def overlay_bounding_boxes(
        self,
        image,
        output_dict,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        line_thickness=4,
        groundtruth_box_visualization_color='black'):
        """Overlay labeled boxes on an image with formatted scores and label names.

        This function groups boxes that correspond to the same location
        and creates a display string for each detection and overlays these
        on the image. Note that this function modifies the image in place, and returns
        that same image.

        Args:
            image: uint8 numpy array with shape (img_height, img_width, 3)
            output_dict: output from object detection algorithm
            category_index: a dict containing category dictionaries (each holding
            category index `id` and category name `name`) keyed by category indices.
            min_score_thresh: minimum score threshold for a box or keypoint to be
            visualized.
            line_thickness: integer (default: 4) controlling line width of the boxes.
            mask_alpha: transparency value between 0 and 1 (default: 0.4).
            groundtruth_box_visualization_color: box color for visualizing groundtruth
            boxes

        Returns:
            uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
        """
        # Create a display string (and color) for every box location, group any boxes
        # that correspond to the same location.
        boxes = output_dict['detection_boxes']
        classes = output_dict['detection_classes']
        scores = output_dict['detection_scores']
        
        box_to_display_str_map = collections.defaultdict(list)
        box_to_color_map = collections.defaultdict(str)
        max_boxes_to_draw = boxes.shape[0]
        for i in range(boxes.shape[0]):
            if max_boxes_to_draw == len(box_to_color_map):
                break
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
                class_name = self.label_map[classes[i]]
                display_str = str(class_name)
                display_str = '{}: {}%'.format(display_str, round(100*scores[i]))
                try:
                    box_to_display_str_map[box].append(display_str)
                except:
                    break
                box_to_color_map[box] = self.STANDARD_COLORS[
                classes[i] % len(self.STANDARD_COLORS)]

        # Draw all boxes onto image.
        for box, color in box_to_color_map.items():
            ymin, xmin, ymax, xmax = box
            image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
            self.draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                                        line_thickness, box_to_display_str_map[box])
            np.copyto(image, np.array(image_pil))

        return image
    
    # Let's perform some estimation for the pixel coordinates of the trash objects


    # Perform object detection
    def detect(self, image, only_trash=False):
        image = np.asarray(image)
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis,...]

        model_fn = self.model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        return output_dict
    
    # Filter for trash items (modify this list based on your requirements)
    def filter_for_trash_items(self, output_dict):
        classes = output_dict['detection_classes']
        trash_indices = [i for i, label in enumerate(classes) if self.label_map[label] \
            in self.trash_labels]
        output_dict_old = output_dict
        output_dict = {'detection_boxes': output_dict_old['detection_boxes'][trash_indices],
                        'detection_classes': output_dict_old['detection_classes'][trash_indices],
                        'detection_scores': output_dict_old['detection_scores'][trash_indices],
                        'num_detections': len(trash_indices)}
        return output_dict
