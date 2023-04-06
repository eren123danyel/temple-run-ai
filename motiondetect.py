# import the necessary packages
import numpy as np
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf

# Load the tf model
mnet = tf.lite.Interpreter(model_path="lite-model_movenet_multipose_lightning_tflite_float16_1.tflite")
mnet.allocate_tensors()

# Function to unnormalize coords
def unnormalize(posx, posy):
    # Get the scale from diving the size of the img we are giving to movenet / frame
    scale = 256 / 640

    # account for padding
    padh = (256 - int(480 * scale)) // 2
    padw = (256 - int(640 * scale)) // 2

    # Calculate new pos
    posx = (posx * 256 - padw) / scale
    posy = (posy * 256 - padh) / scale
    return (int(posx),int(posy))

# Function to process image
def movenet(input_img):
    # Resize to fit model
    input_img = tf.image.resize_with_pad(input_img,256,256)

    # Expand dimension
    input_img = tf.expand_dims(input_img,axis=0)

    # Cast to usigned int
    input_img = tf.cast(input_img,dtype=tf.uint8)

    # Convert to a TensorFlow tensor
    input_tensor = tf.convert_to_tensor(input_img)
    
    # Setup input and output
    input_details = mnet.get_input_details()
    ouput_details = mnet.get_output_details()

    # Is it a dynamic input?
    is_dynamic_shape_model = input_details[0]['shape_signature'][2] == -1
    if is_dynamic_shape_model:
        input_tensor_index = input_details[0]['index']
        input_shape = input_tensor.shape
        mnet.resize_tensor_input(input_tensor_index, input_shape, strict=True)

    # Make prediction
    mnet.allocate_tensors()
    mnet.set_tensor(input_details[0]['index'], input_img.numpy())
    mnet.invoke()
    keypoints_with_scores = mnet.get_tensor(ouput_details[0]['index'])
    return keypoints_with_scores

# Open window
cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # copy frame
    img = frame.copy()
    # run through movenet
    kpoints = movenet(img)[0][0]

    # Get the keypoints for the nose
    nose_y, nose_x, nose_s = kpoints[:3]
    nose = unnormalize(nose_x,nose_y)

    # Left eye
    left_eye_y, left_eye_x, left_eye_s = kpoints[3:6]
    left_eye = unnormalize(left_eye_x, left_eye_y)

    # draw nose
    cv2.circle(frame,nose,2,(0,0,255),-1)

    # draw left eye
    cv2.circle(frame,left_eye,2,(0,0,255),-1)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)