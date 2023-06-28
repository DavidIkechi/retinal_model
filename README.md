# Deep Learning-based Glaucoma Segmentation Model
This project aims to develop a robust segmentation model for detecting glaucoma using a state-of-the-art deep learning methodology. The model is trained on a dataset obtained from RIGA, and the resulting weights for the Optic Cup and Optic Disc are saved.
Furthermore, the weights are loaded and re-used to develop a new model using images acquired from a low-cost ophthalmoscope.

## Languages Used
- Python

## Machine Learning Frameworks and Libraries
- TensorFlow: A popular deep learning framework used for building and training neural networks.
- Keras: A high-level neural networks API that runs on top of TensorFlow, providing an easy-to-use interface for model development.
- OpenCV: A computer vision library used for image processing and manipulation.

## Dataset
The dataset used for training the model is from RIGA, which provides a diverse range of images for accurate glaucoma segmentation. The dataset has been carefully annotated to ensure reliable ground truth for training the model.

## Model Development
The segmentation model is built using Python and the TensorFlow-Keras framework. The code provided includes the necessary steps for loading the dataset, preprocessing the images, building the deep learning model architecture, and training the model using the annotated data.

## Transfer Learning
To further enhance the applicability of the model, the saved weights are loaded and utilized to develop a new model using images acquired from a low-cost ophthalmoscope. This transfer-learning approach allows the model to be effective even with images obtained from different sources or devices.

## Model Evaluation and Testing
To evaluate the performance of the developed model, a set of test images from the MESSIDOR and Arclight datasets are provided. These test images can be used to assess the accuracy and effectiveness of the segmentation model. The code also includes the necessary functions for loading the saved weights, performing segmentation on the test images, and evaluating the results.

## Running the Code
To run the code and utilize the segmentation model, please follow these steps:

- Install the required dependencies: TensorFlow, Keras, and OpenCV.
- Download the RIGA dataset and place it in the designated folder.
- Modify the code as needed to specify the file paths and adjust any hyperparameters.
- Run the code to train the segmentation model on the RIGA dataset.
- Once the model is trained, save the weights for the Optic Cup and Optic Disc.
- Load the saved weights and use the model to perform segmentation on test images from the MESSIDOR and Arclight datasets.
- Evaluate the segmentation results and analyze the performance of the model.

Please refer to the provided Python code and comments for detailed instructions and guidance on each step. Feel free to customize the code and parameters to suit your specific requirements and datasets.








