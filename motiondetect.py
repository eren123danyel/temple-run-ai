import numpy as np
import cv2
import tensorflow as tf
import pyautogui as pg
import collections

# Constants
input_size = 192
width = 640
height = 480
scale = input_size / height
pad_h = (input_size - int(width * scale)) // 2
pad_w = (input_size - int(height * scale)) // 2
last_key_pressed = None

# Load the TensorFlow Lite model
mnet = tf.lite.Interpreter(model_path="lite-model_movenet_multipose_lightning_tflite_float16_1.tflite")
mnet.allocate_tensors()


# Function to unnormalize coordinates
def unnormalize(posx, posy):
    posx = (posx * input_size - pad_w) / scale
    posy = (posy * input_size - pad_h) / scale
    return (int(posx), int(posy))


# Function to process image
def movenet(input_img):
    input_img = tf.image.resize_with_pad(input_img, input_size, input_size)
    input_img = tf.expand_dims(input_img, axis=0)
    input_img = tf.cast(input_img, dtype=tf.uint8)
    input_tensor = tf.convert_to_tensor(input_img)

    input_details = mnet.get_input_details()
    output_details = mnet.get_output_details()

    is_dynamic_shape_model = input_details[0]['shape_signature'][2] == -1
    if is_dynamic_shape_model:
        input_tensor_index = input_details[0]['index']
        input_shape = input_tensor.shape
        mnet.resize_tensor_input(input_tensor_index, input_shape, strict=True)

    mnet.allocate_tensors()
    mnet.set_tensor(input_details[0]['index'], input_img.numpy())
    mnet.invoke()
    keypoints_with_scores = mnet.get_tensor(output_details[0]['index'])

    person_index = np.argmax(keypoints_with_scores[0, :, -1])
    keypoints = keypoints_with_scores[0, person_index]
    keypoints = keypoints[-5:]

    keypoints = [unnormalize(keypoints[1], keypoints[0]), unnormalize(keypoints[3], keypoints[2]), keypoints[4]]
    return keypoints



# Function to determine which key should be pressed
def key_to_press(centerx, centery, threshold=20):
    global last_key_pressed
    
    if centerx <= width / 3 - threshold:
        if last_key_pressed != 'left':
            pg.keyUp('right')
            pg.keyDown('left')
            last_key_pressed = 'left'
    elif centerx >= width / 3 * 2 + threshold:
        if last_key_pressed != 'right':
            pg.keyUp('left')
            pg.keyDown('right')
            last_key_pressed = 'right'
    else:
        if last_key_pressed is not None:
            pg.keyUp('left')
            pg.keyUp('right')
            last_key_pressed = None



# Start the OpenCV window
cv2.startWindowThread()
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', width, height)


# Open webcam video stream
##cv2.CAP_V4L2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


frame_counter = 0 
while True:
    ret, frame = cap.read()
    
    frame_counter += 1

    # Skip every fifth frame
    if frame_counter % 5 != 0:
        continue
    
    keypoints = movenet(frame)

    # Calculate the center
    center = np.mean(keypoints[:2], axis=0).astype(int)

    # Calculate which part of the image you are in
    key_to_press(center[0],center[1])

    # Draw the center
    cv2.circle(frame, center, 5, (255, 0, 0), -1)
    #cv2.rectangle(frame,keypoints[0],keypoints[1],(255,0,0),2)    
    
    # Show frame
    cv2.imshow('frame', frame)

    # Wait 10ms and if q is pressed, break the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()