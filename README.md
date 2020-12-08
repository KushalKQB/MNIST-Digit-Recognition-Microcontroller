# MNIST Digits Recognition on Microcontroller

Neural Networks are memory-hungry in nature. This fact alone constitutes for most of the difficulties in implementing deep learning based applications on resource-constrained devices such as microcontrollers. For instance, a classical neural network with 1 hidden layer and 784 neurons, trained to recognize MNIST digits requires ~2.34 MB of memory for the hidden layer alone. Though it achieves an accuracy of ~98%, general implementations just don't make sense for resource-constrained hardware. This issue, however, can be addressed using various optimization techniques. As we are particularly interested in making a microcontroller recognize a handwritten digit, an optimization technique called 'Dimensionality Reduction' serves the cause extremely well.

Dimensionality reduction is a technique, as in Data Science, used to reduce the size of a given feature. This technique basically simplifies data sets (i.e., reduce the dimensionality) for easier analysis and visualization, but it does so at the cost of some accuracy.

This project demonstrates dimensionality reduction of MNIST digits dataset using Principal Component Analysis (PCA). Instead of training a neural network on the raw image with 784 pixels, this project demonstrates how one can reduce these 784 features down to just 28 (1 principal component out of 28 possible components, each containing 28 features), and still achieve decent accuracy. This (data pre-processing) algorithm reduces the entire size of the required (classical) neural network model dramatically.

## Algorithm and Implementation Details
The implementation mainly involves 2 stages:
1. Dimensionality Reduction and Neural Network Training
2. Dimensionality Reduction and Inferencing on Target

### Dimensionality Reduction and Neural Network Training
Scikit-learn provides the necessary modules required for [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). It is recommended to understand the details of this module and the usage first.

Upon computing the 1st principal component of every image in the dataset, the components are then fed as input to a neural network. (See usage instruction and dive deep into the code for additional info). The first component obtained through PCA corresponds to the most dominant eigenvalue of the image-matrix. In other words, this component contains maximum information of the image-matrix, enough to train a neural network on.

The neural network has (as stated before) 28 inputs, 1 hidden layer with 28 neurons and a output layer with 10 neurons for the number of classes. The hidden layer uses ReLU as its activation function and the output layer uses softmax. The model uses the loss function 'categorical cross-entropy' and 'adam' optimizer.

With 100 epochs, the accuracy achieved is ~81%. With persistent observations, it was found that the loss function reached the local minima after about ~67 epochs. More accuracy can be squeezed out using additional layers the accuracy-memory trade-off is well balanced with 1 hidden layer (guessing 4 right out of 5 digits may not be state-of-the-art but for a model with such minimal footprint, it is not very far).

Now that we have a 'trained' neural network model, we can transfer this model (manually in this project) onto the target microcontroller. The target's job is to only perform inference.

### Dimensionality Reduction and Inferencing on Target
There are no off-the-shelf libraries providing PCA functions in C. Inspired by the guts of the scikit-learn's PCA modules, the equivalent algorithm is implemented in C for the target's environment. The algorithm involves the steps briefed below:
1. Standardize image-matrix
2. Obtain ```(standardized_matrix.T) dot (standardized_matrix)```
3. Use [Power-Iteration](https://en.wikipedia.org/wiki/Power_iteration) method to obtain most dominant eigenvector of the matrix from step 2.
4. Ensure that the highest absolute value in the eigenvector is a positive value.
5. Obtain ```(standardized_matrix) dot (eigenvector_from_step_4)```. This is the 1st and the most dominant component in the input image.

Inferencing in C, for the target's environment, is simple and mainly boils down to basic matrix operations. CMSIS-DSP library simplifies it further with intuitive matrix types and easy-to-use ops. The steps for inferencing (target-independent) are briefed below:
1. Allocate memory for 784 float32 values for the input matrix/image.
2. Allocate memory for 784 float32 values for the hidden layer weights
3. Allocate memory for  28 float32 values for the hidden layer biases.
4. Allocate memory for 280 float32 values for the output layer weights.
5. Allocate memory for  10 float32 values for the output layer biases.
6. Initialize these with the corresponding values from stage 1.
7. Perform PCA and obtain the dominant component.
8. Compute ```(hidden_layer_weights) dot (dominant_component) + (hidden_layer_biases)``` and apply ReLu
9. Compute ```(output_layer_weights) dot (step_8_output) + (output_layer_biases)``` and apply softmax. Predictions obtained in this step.

## Usage

Stage 1:

Requirements: python3, tensorflow, keras, sklearn, numpy, etc.
1. Run 'PCA_featureGen.py' to obtain PCA components of training set and the test set. The components are stored in 'X_train_features.txt' and 'X_test_features.txt' respectively.
2. Run 'MNIST_NN_PCA.py' to train the neural network model on the results from step 1. Weights and Biases are stored in .txt output files.

Stage 2:

Requirements: CMSIS_DSP lib, STM32CubeIDE for working example
1. Use the skeleton code, containing tested functions for PCA and Inferencing on the target. Please not that the random generator function is STM32 HAL specific. Interested parties must change this. Do not compile this file; it's more of a code-handout.
2. For a working example, download the STM32CubeIDE based project for STM32F4 Discovery Board.

For further details, please dive into the code. The documentation should suffice any other missing implementational details here.

## Note
This is a proof-of-concept. Please treat it like one. A more github-like repo is in the works which would be much more plug-n-play friendly.
