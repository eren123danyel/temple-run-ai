import numpy as np
import cv2
import tensorflow as tf
import pyautogui as pg

# Constants
input_size = 192
width = 480
height = 640
scale = input_size / height
pad_h = (input_size - int(width * scale)) // 2
pad_w = (input_size - int(height * scale)) // 2

# Load the TensorFlow Lite model
mnet = tf.lite.Interpreter(model_path="lite-model_movenet_multipose_lightning_tflite_float16_1.tflite",num_threads=4)
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
def key_to_press(centerx,centery):    
    if centerx <= width / 3:
        print("left")
    elif centerx <= width / 3 * 2:
        print("middle")
    else:
        print("right")



# Start the OpenCV window
cv2.startWindowThread()


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
    if frame_counter % 5 == 0:
        continue
    
    keypoints = movenet(frame)

    # Calculate the center
    center = np.mean(keypoints[:2], axis=0).astype(int)

    # Calculate which part of the image you are in
    key_to_press(center[0],center[1])
    # Draw bounding box
    cv2.circle(frame, center, 5, (255, 0, 0), -1)
    
    # Show frame
    cv2.imshow('frame', frame)

    # If q is pressed, break the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
