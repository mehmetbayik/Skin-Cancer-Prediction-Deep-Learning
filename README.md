# CS464 Introduction to Machine Learning
Assignment and project implementations done for CS464 course, Bilkent University, Fall 2023.

## Coverage of the repository

### HW1
Probability Review and Naive Bayes

### HW2
PCA, Linear Regression, Logistic Regression, and SVM.

### HW3
Convolutional Neural Networks using PyTorch.

## Project

Skin cancer is one of the most common cancers globally and early detection plays a crucial role in improving patients’ prognosis. Accurate classification of dermatoscopic images into specific categories is important for medical diagnosis and treatment planning. This project will be able to automate skin cancer classification process and will reduce the inspection time. While this project cannot replace clinical inspection, it can be a useful tool for preliminary examination of the patients.

### Dataset Description

The dataset is available on harvard.edu website. Dataset contains 10015 dermatoscopic images with a total of 7 classes. These classes are different diagnostic categories for pigmented lesions: actinic keratoses and intraepithelial carcinoma (akiec), basal cell carcinoma (bcc), benign keratosis-like legions (bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv), and vascular lesions (vasc). The dataset also contains features like age, gender, and location of the cancer. These features can be used to increase the accuracy of results. The category distribution is not balanced and nv category contains 67% of the images. This might require data augmentation for the other categories.

Since this is an old competition, test images are also available. And labels for the test images are available on the website since Feb 7, 2023. This will extend our dataset from 10015 images to 11525 images.

Since our test set is given, we will use test set only for model comparison at the end. We split our other data to training and validation sets to train our models and tune hyperparameters.

Dataset link:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

### Project Plan

We plan to use PyTorch and TensorFlow as the main machine learning frameworks in this project. We plan to use multiple neural networks (NN) for comparison as this is an image processing task. The first NN is Convolutional Neural Networks (CNN) since it is highly used in image classification tasks. CNN can reduce images into a form which is easier to process without losing features [2]. The second will be Residual Neural Networks (ResNet). ResNet has more trainable parameters and allows for better feature capturing, which makes it better for image classification. We are also planning to use Transfer Learning for comparison purposes and see how well our model has done.
