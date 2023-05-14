# Face Mask Detection using InceptionResNetV2

This project aims to detect whether a person is wearing a face mask or not using the InceptionResNetV2 deep learning model. The model is trained on a dataset of images labeled with and without face masks.

## Introduction

Face mask detection has become crucial in the context of the COVID-19 pandemic. This project uses computer vision techniques and deep learning to analyze images and classify whether a person is wearing a face mask or not. The InceptionResNetV2 model, pre-trained on ImageNet, is fine-tuned on a custom dataset for accurate mask detection.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/face-mask-detection.git
   ```
2. Download the pre-trained weights for the InceptionResNetV2 model:

[Pre-trained weights](https://drive.google.com/file/d/1Ho0KnSNAb2nlbxOSxp0MYm0xJfdk9MkN/view?usp=share_link)



## Usage

1. Rename all the files in the dataset if you want
   ```bash
   python rename.py
   ```
2. Train the model if you want
   ```bash
   python main.py
   ```
3. Run the real_time.py script to open the real-time face mask detection application:
   ```bash
   python real_time.py
   ```
   
## Dataset

The dataset used for training and evaluation can be downloaded from the following link:

[Dataset](https://universe.roboflow.com/pyimagesearch/covid-19-pis/dataset/2)

The dataset contains images of people wearing and not wearing face masks.

## Model Training
The InceptionResNetV2 model is used as the base model for face mask detection. The pre-trained weights on ImageNet are loaded, and the last few layers are fine-tuned on the custom dataset. The model is trained using the Adam optimizer and sparse categorical cross-entropy loss.

## Results
The trained model achieves an accuracy of 95% on the validation set. The performance may vary based on the quality of input images and real-world conditions.

## Contributing
Contributions are welcome! If you would like to contribute to this project, please follow these steps:

Fork the repository.
Create a new branch for your feature or bug fix.
Make your changes and commit them.
Push your changes to your forked repository.
Submit a pull request explaining your changes.
