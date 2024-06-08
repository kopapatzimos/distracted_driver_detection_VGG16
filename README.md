Abstract

The distracted driver detection model is thoroughly examined in this analytical documentation, with particular attention paid to the model's design, training process, implementation specifics, and possible enhancements. The model makes use of pre-trained convolutional neural network (CNN) architectures, including VGG16, and fine-tuning for enhanced performance, in addition to transfer learning approaches.

Introduction
Road safety is seriously threatened by distracted driving, which is a major cause of accidents and fatalities across the globe. In order to solve this problem, this research builds a system that can automatically identify distracted driving behaviours from photos using deep learning techniques. The State Farm Distracted Driver Detection competition on Kaggle is the source of the dataset used in this project, which offers an extensive collection of photos representing different driving situations.


VGG16	
Every convolutional block in VGG16 is composed of several 3x3 convolutional layers, followed by a max-pooling layer, giving rise to its uniform design. The network may be easily scaled and interpreted thanks to this homogeneity. Furthermore, VGG16 makes use of deeper layers and tiny filter sizes (3x3), which let it successfully capture complex patterns and hierarchical characteristics in images.


Transfer learning involves leveraging pre-trained models trained on large datasets and adapting them to new tasks with smaller datasets. VGG16 is an excellent candidate for transfer learning due to several reasons:
Availability: Pre-trained VGG16 models are readily available in popular deep learning libraries such as TensorFlow and Keras, making it convenient for practitioners to use.
Generalization: VGG16 has demonstrated strong generalization capabilities across various image classification tasks. Its learned features are often applicable to a wide range of visual recognition tasks.
Simplicity: VGG16's straightforward architecture makes it easy to understand and modify, making it suitable for adaptation to specific tasks or datasets.
Performance: Despite its simplicity, VGG16 achieves competitive performance on benchmark datasets like ImageNet. Its deeper architecture allows it to capture more complex features, leading to higher classification accuracy.
Project Workflow
The project workflow encompasses several key steps:
Data Extraction: The dataset is obtained from the State Farm Distracted Driver Detection competition on Kaggle, consisting of images captured from in-car cameras depicting various driving activities.
Data Preparation: The dataset is preprocessed and organized into training, validation, and testing sets. Additionally, data augmentation techniques are employed to enhance the diversity and robustness of the training data.
Model Development: Deep learning models, including pre-trained architectures such as VGG16 and custom-built models, are developed to classify images into different distracted driving behaviors.
Training and Evaluation: The models are trained using the prepared dataset, and their performance is evaluated on unseen validation and test sets. Metrics such as accuracy, precision, recall, and F1-score are used to assess model performance.
Fine-Tuning and Optimization: Techniques such as fine-tuning and hyperparameter optimization are applied to improve model performance further. This involves adjusting model architecture, learning rates, and regularization techniques.
Deployment and Integration: Once the model achieves satisfactory performance, it can be deployed in real-world settings, integrated into existing automotive systems or mobile applications for real-time distracted driving detection.


Step 1: Data Extraction from Kaggle
In this phase, we use the Google Colab environment's Kaggle API to download the dataset straight from Kaggle. Convenience, reproducibility, and smooth interaction with the Colab environment are only a few benefits of this method.
1. Setting up Kaggle API
First, we install the Kaggle API package using pip, which allows us to interact with Kaggle directly from our Python environment:

!pip install -q kaggle



Next, we upload our Kaggle API credentials file (kaggle.json) to authenticate our access to the Kaggle platform:
from google.colab import files
files.upload()


The kaggle.json file contains our Kaggle username and API key, which are required for authentication.
2. Downloading the Dataset
With the Kaggle API set up, we can now use it to download the dataset from Kaggle. We use the kaggle datasets download command, providing the dataset's unique identifier (competition slug or dataset name) obtained from the Kaggle website:

!kaggle competitions download -c state-farm-distracted-driver-detection

This command downloads the dataset files as a zip archive directly to our Colab environment.
3. Extracting the Dataset
Once the dataset zip file is downloaded, we can extract its contents using the unzip command:

!unzip state-farm-distracted-driver-detection




Step 2: Data Preparation
In this step, we organize the downloaded dataset into training and validation sets to facilitate model training and evaluation. This involves partitioning the dataset into two subsets: one for training the model and the other for validating its performance.
1. Directory Structure
We begin by defining a directory structure for our dataset, typically consisting of separate directories for training, validation, and testing data. This structured organization helps in managing the dataset efficiently and ensures consistency during model development.
2. Splitting the Dataset
Next, we split the dataset into training and validation sets. A common approach is to reserve a certain percentage of the data for validation, while the rest is used for training. This partitioning helps assess the model's performance on unseen data and guards against overfitting.
3. Data Augmentation
To enhance the diversity and generalization ability of our model, we may employ data augmentation techniques. These techniques involve applying transformations such as rotation, flipping, and scaling to the training images, thereby generating additional training samples without collecting new data.
4. Directory Manipulation
Once the dataset is split and augmented (if applicable), we manipulate the directory structure accordingly. We create separate directories for training and validation data, and move a portion of the training images into the validation directory.


Step 4: Model Definition and Training
In this step, we define, compile, and train a Convolutional Neural Network (CNN) model based on the VGG16 architecture for the task of distracted driver detection. The VGG16 architecture is chosen for its effectiveness in image classification tasks and its widespread use in various 
computer vision applications.
1. Models
1. Base Model
We insert the VGG16 model excluding the fully connected layers at the top. By leveraging pre-trained weights, the model already possesses knowledge of low-level features, which can accelerate training and improve performance, especially when dealing with limited labeled data.



From the graphs it's obvious that we have succeeded in having low training and validation accuracy.
2. Fine-tuning
In order to achieve better results, we fine-tune the model  by adding custom fully connected layers on top and adjusting the model's parameters to adapt it to our specific task of distracted driver detection. Fine-tuning involves training the model on our dataset while allowing the weights of certain layers, typically the later convolutional layers, to be updated during training.


As we obsever in the graphs, we have achieved greater validation accuracy, however it seems that if we had more epochs the validation accuracy is going to geting more and more smaller.

3. Improved Fine Tuning Model 
The improved model employs the following enhancements:
GlobalAveragePooling2D: Instead of Flatten, GlobalAveragePooling2D is used. This reduces each feature map to a single value by averaging, which helps in reducing the number of parameters and improving generalization.
Regularization: L2 regularization is added to the dense layers to penalize large weights, thus preventing overfitting.
Fine-Tuning: Initially, all layers of VGG16 are frozen, but during fine-tuning, the last few layers are unfrozen to allow the network to adjust high-level features for the specific task.
Dropout Layers: Dropout layers are used extensively to prevent overfitting by randomly dropping neurons during training.



4. Feature Extraction Model 

Next we focus on the feature extraction model with no fine tuning.Feature extraction is a crucial process in machine learning and computer vision that involves transforming raw data into a set of features that can be effectively used by a model for classification, regression, or other tasks. The goal is to capture the most relevant information from the raw data while reducing its dimensionality and noise.




Making the model simpler helped us improve both training and validation accuracy.Base on this we are going to create the final mode, which contains fine tuning and feature extraction to compare our results.


5. Final Model



In the final model  we first trained the model in a feature extraction mode, then fine-tuned it by unfreezing some of the convolutional layers of the VGG16 base model to improve performance on the specific dataset.
we achieved higher accuracy, so this is the model we are going to use for predictions.

2. Model Compilation
We compile the model using an appropriate optimizer, loss function, and evaluation metrics. For example, we may use the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric. The choice of optimizer and loss function depends on the nature of the task and the characteristics of the dataset.
3. Training
The compiled model is then trained on the training dataset. During training, the model learns to map input images to their corresponding distracted driver classes. We typically train the model for 10 epochs, monitoring its performance on a separate validation dataset to prevent overfitting.




Predictions
In order to do the prediction, we created test_ds, loading the test images with the preprocessing that has been applied to train images.
The user has to give the path from the test images in order to create the prediction.



Conclusion
The project successfully developed a distracted driver detection system using the VGG16 architecture and transfer learning techniques. By systematically fine-tuning and optimizing the model, it achieved high accuracy in classifying various distracted driving behaviors. The final model, which incorporates both feature extraction and fine-tuning, demonstrated superior performance, making it suitable for mobile applications for real-time distracted driving detection.
