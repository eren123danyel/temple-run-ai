import numpy as np
import cv2
import tensorflow as tf
import pyautogui as pg

# Constants
input_size = 192

# Load the TensorFlow Lite model
mnet = tf.lite.Interpreter(model_path="lite-model_movenet_multipose_lightning_tflite_float16_1.tflite")
mnet.allocate_tensors()


# Function to unnormalize coordinates
def unnormalize(posx, posy):
    scale = input_size / 640
    padh = (input_size - int(480 * scale)) // 2
    padw = (input_size - int(640 * scale)) // 2

    posx = (posx * input_size - padw) / scale
    posy = (posy * input_size - padh) / scale
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
def key_to_press(xmin, ymin, xmax, ymax):
    centerx = xmin + ((xmax - xmin) / 2)
    centery = ymin + ((ymax - ymin) / 2)
    
    if centerx <= 640 / 3:
        print("left")
    elif centerx <= 640 / 3 * 2:
        print("middle")
    else:
        print("right")

    return (int(centerx), int(centery))

# Start the OpenCV window
cv2.startWindowThread()

# Open webcam video stream
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

while True:
    ret, frame = cap.read()
    img = frame.copy()
    keypoints = movenet(img)

    cv2.rectangle(frame, keypoints[0], keypoints[1], (255, 0, 0), 3)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
