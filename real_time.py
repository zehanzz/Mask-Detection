import cv2
import numpy as np
from keras.applications import InceptionResNetV2
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.image_utils import img_to_array

num_classes = 2

base_model = InceptionResNetV2(include_top=False, pooling='avg', weights='imagenet')

model = Sequential()
model.add(base_model)  # Add the selected pre-trained model as the base
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.load_weights('model_weights.h5')


video_capture = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# cv2.namedWindow('Face Mask Detection', cv2.WINDOW_NORMAL)

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize to the input shape of the model
    img = img_to_array(img)  # Convert to numpy array
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    img = img / 255.0  # Normalize the pixel values
    return img

while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess the face image
        preprocessed_face = preprocess_image(face_roi)

        # Make predictions using the trained model
        prediction = model.predict(preprocessed_face)
        mask_label = 'With Mask' if prediction[0][1] > 0.9188 else 'Without Mask'
        print(prediction[0][0])
        print(prediction[0][1])
        # Draw a rectangle and label around the face
        color = (0, 255, 0) if mask_label == 'With Mask' else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, mask_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame with face detection and mask prediction
    cv2.imshow('Face Mask Detection', frame)
    cv2.waitKey(1)
    # Break the loop
