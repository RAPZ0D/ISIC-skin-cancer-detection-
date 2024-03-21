# ISIC-skin-cancer-detection-
This is a skin cancer detection model, created by using the data from ISIC (The International Skin Imaging Collaboration​) 
If you want to check out their official page here is the link to their website: 
[ISIC Website](https://www.isic-archive.com/)

## ISIC DATASET 
What is ISIC? 
The International Skin Imaging Collaboration, the ISIC Archive is a repository of dermatoscopic images that researchers and developers can use for training and evaluating algorithms for skin cancer detection. The dataset contains images of various skin lesions, including benign and malignant lesions. Each image is typically associated with metadata, such as diagnostic information and lesion characteristics.
This dataset is free and open source, researchers often use datasets like the ISIC Archive to train machine learning models, including Convolutional Neural Networks (CNNs), for automated skin cancer diagnosis. These models can assist dermatologists in early detection and classification of skin lesions.

# Skin Cancer Detection using Convolutional Neural Networks in Python Programming Language 
Convolutional Neural Networks (CNNs) are a powerful class of deep learning models widely used for image analysis tasks like skin cancer detection. In Python, frameworks like TensorFlow, Keras, and PyTorch provide efficient tools for building and training CNN models. CNNs automatically learn features from raw image data, making them well-suited for tasks where manual feature extraction is challenging. With their ability to capture spatial hierarchies and leverage transfer learning, CNNs offer a promising approach for accurate and reliable skin cancer detection.

## Introduction
I'm a data science and AI student who got excited about Convolutional Neural Networks (CNNs). While learning about CNNs, I found the ISIC website. It's a great place with lots of free data and challenges about skin problems. This got me thinking: why not use CNNs to help detect skin cancer?

I was inspired by all the resources on the ISIC website and decided to give it a try. My goal is simple: to use CNNs to tell if a skin lesion is harmful (malignant) or not (benign). This project is my way of learning more about CNNs and doing something meaningful at the same time. I wanted to make a simple but good model I hope to learn more about CNNs and also help fight skin cancer along the way.

## Methodology 
This methodology is an explanation of whay I have done in my Convolution Neural Network
- **Initialising the CNN**:
  - Create a Sequential model, which allows you to stack layers one after the other.
  
- **Step 1 - Convolution**:
  - Add a Conv2D layer with 32 filters, a kernel size of 3x3, and ReLU activation function.
  - Specify the input shape as (128, 128, 3) for images with a resolution of 128x128 pixels and 3 color channels (RGB).

- **Step 2 - Pooling**:
  - Add a MaxPooling2D layer with a pool size of 2x2 and strides of 2 to downsample the feature maps.

- **Adding a second convolutional layer**:
  - Add another Conv2D layer with similar specifications as the first convolutional layer.
  - Follow it with another MaxPooling2D layer for further downsampling.

- **Step 3 - Flattening**:
  - Flatten the 3D feature maps into a 1D vector to feed into the fully connected layers.

- **Step 4 - Full Connection**:
  - Add a Dense layer with 128 units and ReLU activation function.

- **Step 5 - Output Layer**:
  - Add the output layer with 1 unit and a sigmoid activation function for binary classification (malignant or benign).

- **Compiling the CNN**:
  - Compile the model using the Adam optimizer and binary crossentropy loss function for binary classification.
  - Specify 'accuracy' as the metric to monitor during training.

## Results

- **Training Accuracy**:
  - Gradually increased from approximately 53.67% to 95.00%

- **Validation Accuracy**:
  - Fluctuated between approximately 50.00% and 91.00%

- **Training Loss**:
  - Decreased from approximately 1.07 to 0.13

- **Validation Loss**:
  - Varied between approximately 0.25 and 0.99

While the model achieved high accuracy on the training set, there seems to be some overfitting as the validation accuracy fluctuated. This indicates that the model may benefit from further hyperparameter tuning to generalize better to unseen data. However, even with these results, the model shows promise in distinguishing between malignant and benign skin lesions.

## References

- [ISIC Skin Lesion Analysis Towards Melanoma Detection](https://challenge.isic-archive.com/data/)

- [ISIC - International Skin Imaging Collaboration](https://www.isic-archive.com/)

- Tschandl, P., Wurm, M., et al. (2021). Human vs. Artificial Intelligence in Dermatology: A Comparative Study in the Detection of Malignant Melanoma on Clinical Images and Melanocytic Lesions in Histopathological Images. *Journal of Investigative Dermatology*, 141(10), 2307-2316. DOI: [10.1016/j.jid.2021.02.030](https://www.sciencedirect.com/science/article/pii/S1361841521003509)
