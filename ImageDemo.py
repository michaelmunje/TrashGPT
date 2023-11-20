import cv2
import numpy as np
import TrashObjectDetector
import TrashGPT
import TrashMatch
import base64
import sys


def image_demo(filename, save_images=True):
    image = cv2.imread(image_filepath)
    
    trash_detector_gpt = TrashGPT.TrashGPT()
    trash_detector_od = TrashObjectDetector.TrashObjectDetector()
    trash_match = TrashMatch.TrashMatch()

    result = trash_detector_od.detect(image)
    
    if save_images:
        image_copy = image.copy()
        od_img = trash_detector_od.overlay_bounding_boxes(image_copy, result)
        # save od_img
        new_filename = filename.split('.')[0] + '_OD.jpg'
        cv2.imwrite(new_filename, od_img)
        print('Saved image to', new_filename)

    trash_result = trash_detector_od.filter_for_trash_items(result)

    if save_images:
        image_copy = image.copy()
        od_trash_img = trash_detector_od.overlay_bounding_boxes(image_copy, trash_result)
        # save od_img
        new_filename = filename.split('.')[0] + '_OD_Trash.jpg'
        cv2.imwrite(new_filename, od_trash_img)
        print('Saved image to', new_filename)

    # Function to encode the image
    with open(filename, "rb") as image_file:
        gpt_img_input = base64.b64encode(image_file.read()).decode('utf-8')
    
    n_runs = 5
    avg_trash = trash_detector_gpt.perform_trash_detection(gpt_img_input, image.shape[0], image.shape[1], n_runs)
    runs = trash_detector_gpt.raw_runs
    
    if save_images:
        # add another part below the image that has a white background
        # make an img that just white
        # get width of img with height of 200
        
        for run_idx in range(len(runs)):
            if runs[run_idx] is None:
                trash = TrashGPT.Trash('None', [0, 0])
            else:
                trash = runs[run_idx]
        
            width = image.shape[1]
            
            lines = ['GPT-4V:']
            
            if trash.name is not None:
                # convert item.location to relative coordinates
                x = round(trash.location[1] / width, 2)
                y = round(trash.location[0] / image.shape[0], 2)
                lines.append(f'{trash.name}: {y}, {x}')
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
            gpt_image = trash_detector_gpt.overlay_results(image, trash)
            # Add images together
            gpt_image = np.concatenate((gpt_image, caption_img), axis=0)

            # same thing but above
            prompt_img = np.ones((250, width, 3), np.uint8) * 255
            
            prompt_and_example = trash_detector_gpt.trash_prompt.split('\n')
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

            gpt_image = np.concatenate((prompt_img, gpt_image), axis=0)

            # save img
            new_filename = filename.split('.')[0] + f'_GPT{run_idx}.jpg'
            cv2.imwrite(new_filename, gpt_image)
            print('Saved image to', new_filename)
    
    # finally, let's do the matching
    match_result = trash_match.match_results(avg_trash, trash_result, image.shape[0], image.shape[1])
    
    if save_images:
        match_image = trash_detector_gpt.overlay_results(image, match_result)
        # save img
        new_filename = filename.split('.')[0] + '_Match.jpg'
        cv2.imwrite(new_filename, match_image)
        print('Saved image to', new_filename)    


if __name__ == '__main__':
    image_filepath = sys.argv[1]
    
    image_demo(image_filepath)