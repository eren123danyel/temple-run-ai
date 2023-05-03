import cv2
import os
import numpy as np
import tensorflow as tf
import pyautogui
import http.server
from threading import Thread


# make sure we are in the current dir
os.chdir('.')
# Set failsafe to false
pyautogui.FAILSAFE = False

# Load the model using the TensorFlow Lite interpreter
interpreter = tf.lite.Interpreter(model_path="lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite")
interpreter.allocate_tensors()

# Initialize the webcam using OpenCV
cap = cv2.VideoCapture(1)
WIDTH = 480
HEIGHT = 640
INPUT_SIZE = 192
SCALE = INPUT_SIZE / HEIGHT
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
PAD_H = (INPUT_SIZE - int(WIDTH * SCALE)) // 2
PAD_W = (INPUT_SIZE - int(HEIGHT * SCALE)) // 2
last_key_pressed = None


def unnormalize(pos):
    return (int((pos[0] * INPUT_SIZE - PAD_H) / SCALE), int((pos[1] * INPUT_SIZE - PAD_W) / SCALE))

def main():
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Resize and pad the frame to 192x192
        resized_frame = tf.image.resize_with_pad(frame, INPUT_SIZE, INPUT_SIZE)

        # Convert the frame to a float32 NumPy array
        input_data = np.expand_dims(resized_frame, axis=0)
        input_data = tf.cast(input_data, dtype=tf.uint8)

        # Set the input tensor for the TensorFlow Lite interpreter
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]['index'], input_data.numpy())

        # Invoke the interpreter to perform pose estimation on the input frame
        interpreter.invoke()

        # Extract the pose keypoints with scores from the output tensor of the interpreter
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Display the pose keypoints on the original frame using OpenCV
        keypoints = output_data[0, 0, :, :2]
        scores = output_data[0, 0, :, 2]
        left_shoulder,right_shoulder = keypoints[5], keypoints[6]
        left_ankle, right_ankle = keypoints[15],keypoints[16]
        
        if scores[5] > 0.3 and scores[6] > 0.3:
            # Unnormalize coords
            y_left, x_left = unnormalize(left_shoulder)
            y_right, x_right = unnormalize(right_shoulder)
            x_mid = (x_left + x_right) // 2
            y_mid = (y_left + y_right) // 2
            
            cv2.circle(frame,(int(x_mid),int(y_mid)),5,(255,0,0),-1)

            # Press and hold the left arrow key if the person is on the right  side of the screen
            if x_mid < frame.shape[1] // 3:
                last_key_pressed = "right"
            # Press and hold the right arrow key if the person is on the left  side of the screen
            elif x_mid > 2 * frame.shape[1] // 3:
                last_key_pressed = "left"
            # Release both arrow keys if the person is in the middle of the screen
            else:
                last_key_pressed = None

            if last_key_pressed == "left":
                pyautogui.keyDown('left')
                pyautogui.keyUp('right')
            
            elif last_key_pressed == "right":
                pyautogui.keyDown('right')
                pyautogui.keyUp('left')
            else:
                pyautogui.keyUp('left')
                pyautogui.keyUp('right')

            # Ducking logic: Press the down arrow key if person is below half of the screen
            if y_mid > HEIGHT/2:
                pyautogui.press('down')

        ## Jumping logic
        if scores[15] > 0.3 and scores[16] > 0.3:
            # Unnormalize coords
            y_left, x_left = unnormalize(left_ankle)
            y_right, x_right = unnormalize(right_ankle)
            y_mid = (y_left + y_right) // 2
            # Press the up arrow key if the person is jumping 
            if y_mid < HEIGHT-230:
                pyautogui.press('up')
    

        
        # Show the frame with pose keypoints
        cv2.imshow('frame', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Release the webcam and close all OpenCV windows
            cap.release()
            cv2.destroyAllWindows()
            os._exit()



if __name__ == "__main__":
    webServer = http.server.HTTPServer(server_address=('', 8000), RequestHandlerClass=http.server.CGIHTTPRequestHandler)
    print("Website running on localhost:8000")
    Thread(target = webServer.serve_forever).start()
    Thread(target = main).start()
    
    
    
   



