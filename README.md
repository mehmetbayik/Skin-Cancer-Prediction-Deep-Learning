# CS464 Introduction to Machine Learning
Assignment and project implementations done for CS464 course, Bilkent University, Fall 2023.

## Coverage of the repository

### HW1
Three different Naive Bayes Classifier implemented using Python, Numpy and Pandas without using any ML specific libraries.

### HW2
PCA and Multinomial Logistic Regression implemented using Python, Numpy and Pandas without using any ML specific libraries.

### HW3
Decoder-Encoder based image reconstruction with CNN using PyTorch.

## Project

Skin Cancer Prediction using Deep Learning algorithms

* CNN (Convolutional Neural Networks) implemented using TensorFlow and Keras.

* ResNet34 (Residual Networks) imitation implemented using PyTorch.

* Transfer Learning with Resnet50 and VGG19 implemented using Tensorflow.

* Additionaly, XGBoost and Random Forest implemented using Scikit-learn after feature extraction with CNN.

All of the models are trained, optimized and compared on the dataset below.


### Dataset Description

HAM10000 dataset is available on harvard.edu website. Dataset contains 10015 dermatoscopic images with a total of 7 classes. 

These classes are different diagnostic categories for pigmented lesions: actinic keratoses and intraepithelial carcinoma (akiec), basal cell carcinoma (bcc), benign keratosis-like legions (bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv), and vascular lesions (vasc). 

The dataset also contains features like age, gender, and location of the cancer. These features is used to increase the accuracy of results. The category distribution is not balanced and nv category contains 67% of the images. This might require data augmentation for the other categories.

Since this is an old competition, test images are also available. And labels for the test images are available on the website since Feb 7, 2023. This will extend our dataset from 10015 images to 11525 images.

Since our test set is given, we used fixed test set only for model comparison at the end. We split our other data to training and validation sets to train our models and tune hyperparameters.

[Dataset link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

### License

[MIT](https://choosealicense.com/licenses/mit/)
