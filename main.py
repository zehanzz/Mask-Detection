import os
import cv2
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras import Model
from keras.applications import InceptionResNetV2
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

with_masks_folder = 'dataset/train/with_mask'
without_masks_folder = 'dataset/train/without_mask'

# Define the desired image size for resizing
image_size = (224, 224)  # Adjust as needed

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize empty lists for storing the preprocessed images and corresponding labels
images = []
labels = []

# Preprocess images with masks
with_masks_files = os.listdir(with_masks_folder)
for filename in with_masks_files:
    if filename.endswith(".jpg"):
        image_path = os.path.join(with_masks_folder, filename)
        print(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over the detected faces
        for (x, y, w, h) in faces:
            # Extract the face region from the image
            face_roi = image[y:y + h, x:x + w]

            # Resize the face region to the desired size
            resized_face = cv2.resize(face_roi, image_size)

            # Append the resized face to the images list
            images.append(resized_face)
            labels.append(1)  # Label 1 for images with masks

# Preprocess images without masks
without_masks_files = os.listdir(without_masks_folder)
for filename in without_masks_files:
    if filename.endswith(".jpg"):
        image_path = os.path.join(without_masks_folder, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over the detected faces
        for (x, y, w, h) in faces:
            # Extract the face region from the image
            face_roi = image[y:y + h, x:x + w]

            # Resize the face region to the desired size
            resized_face = cv2.resize(face_roi, image_size)

            # Append the resized face to the images list
            images.append(resized_face)
            labels.append(0)  # Label 1 for images with masks

# Convert the lists to NumPy arrays for further processing
images = np.array(images)
labels = np.array(labels)
#
# # Verify the shapes of the arrays
print('Images shape:', images.shape)
print('Labels shape:', labels.shape)

# Split the dataset into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42)

# Verify the shapes of the split datasets
print('Train images shape:', train_images.shape)
print('Train labels shape:', train_labels.shape)
print('Validation images shape:', val_images.shape)
print('Validation labels shape:', val_labels.shape)

data_augmentation = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

base_model = InceptionResNetV2(include_top=False, pooling='avg', weights='imagenet')

# Define the number of classes
num_classes = 2  # Number of classes (with mask and without mask)

# Build the model
model = Sequential()
model.add(base_model)  # Add the selected pre-trained model as the base
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Define the path to save the model weights
weights_filepath = 'model_weights.h5'

# Create a ModelCheckpoint callback to save the best model weights
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', save_best_only=True)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Define training parameters
batch_size = 32
epochs = 100

train_data_generator = data_augmentation.flow(train_images, train_labels, batch_size=batch_size)

# Train the model
history = model.fit(train_data_generator,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(val_images, val_labels),
                    callbacks=[checkpoint])

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(val_images, val_labels, verbose=1)

# Print the evaluation results
print('Validation Loss:', loss)
print('Validation Accuracy:', accuracy)

model.save_weights(weights_filepath)

training_loss = history.history['loss']
training_accuracy = history.history['accuracy']
validation_loss = history.history['val_loss']
validation_accuracy = history.history['val_accuracy']

# Create a plot for the loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), training_loss, label='Training Loss')
plt.plot(range(1, epochs+1), validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()

# Create a plot for the accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), training_accuracy, label='Training Accuracy')
plt.plot(range(1, epochs+1), validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()






# def preprocess_image(input_image):
#     # Convert the input image to grayscale
#     if input_image.ndim == 2:
#         # If the input image has only one channel, assume it is already grayscale
#         gray = input_image
#     elif input_image.ndim == 3:
#         # If the input image has three channels, convert it to grayscale
#         gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
#     else:
#         # Handle the case when the input image has an unexpected number of channels
#         raise ValueError("Input image has an unsupported number of channels.")
#
#     # Normalize the pixel values to a range of 0 to 1
#     normalized_image = gray / 255.0
#
#     # Resize the image to a suitable size
#     resized_image = cv2.resize(normalized_image, (224, 224))
#
#     return resized_image
#
#
# # Define the face cascade classifier
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# # Define the mask threshold
# mask_threshold = 0.5
#
# # Initialize the video capture from the default camera
# video_capture = cv2.VideoCapture(0)
#
# while True:
#     # Read a frame from the video capture
#     ret, frame = video_capture.read()
#
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces in the grayscale frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     # Iterate over the detected faces
#     for (x, y, w, h) in faces:
#         # Extract the face region from the frame
#         face_roi = gray[y:y + h, x:x + w]
#
#         # Preprocess the face region
#         preprocessed_face = preprocess_image(face_roi)
#
#         # Pass the preprocessed face through the face recognition model to obtain predictions
#         predictions = model.predict(np.expand_dims(preprocessed_face, axis=0))
#
#         # Get the predicted class and confidence
#         predicted_class = np.argmax(predictions[0])
#         confidence = predictions[0][predicted_class]
#
#         # Check if the face is not wearing a mask based on the confidence score
#         if predicted_class == 0 and confidence > mask_threshold:
#             # Draw a red rectangle around the face to indicate no mask
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#     # Display the frame with real-time face detection
#     cv2.imshow('Real-time Face Detection', frame)
#
#     # Exit the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture
# video_capture.release()
#
# # Close any OpenCV windows
# cv2.destroyAllWindows()