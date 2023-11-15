import cv2
import time
import numpy as np
import TrashObjectDetector
import TrashGPT
import base64


if __name__ == '__main__':
    trash_detector_gpt = TrashGPT.TrashGPT()
    trash_detector_od = TrashObjectDetector.TrashObjectDetector()

    # when you press Q, a picture is taken and queried to OD and GPT
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
                
                trash = trash_detector_gpt.perform_trash_detection(image, frame.shape[0], frame.shape[1])
                
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
