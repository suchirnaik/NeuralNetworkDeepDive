{\rtf1\ansi\ansicpg1252\cocoartf2706
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Neural Network Exploration Project: A Deep Dive into Machine Learning\
\
## Introduction\
\
Hello and welcome to my detailed exploration into the world of neural networks! This project is my personal journey through the complexities and wonders of machine learning. It covers everything from foundational theory to practical application, showcasing my growth and learning in the field.\
\
## Project Layout\
\
This README details a multi-part project, each focusing on different aspects of neural network applications and theory. From theoretical calculations to real-world applications, this journey is both extensive and intensive.\
\
## Part I: Problem Solving\
\
**1. Neural Network Topology and Calculation**\
\
**Objective**: Understanding the topology of a neural network and performing manual calculations.\
\
**Tasks**:\
- **(a) Output Generation**: Calculated the outputs of a feedforward neural network with two input neurons, two hidden neurons, and two output neurons using ReLU activation functions. Manually determined the \uc0\u55349 \u56475  and \u55349 \u56450  values for each neuron.\
- **(b) Sum of Squared Errors (SSE)**: Computed the SSE for the network to assess the difference between actual and desired outputs.\
- **(c) Error Delta (\uc0\u55349 \u57145 )**: Calculated the error delta for each processing neuron (\u55349 \u57087 6,\u55349 \u57087 5,\u55349 \u57087 4,\u55349 \u57087 3) in the network.\
- **(d) Sensitivity of Error to Weights**: Determined how changes in each weight affect the network's overall error.\
- **(e) Weight Update**: Updated the network weights assuming a learning rate (\uc0\u55349 \u57142 ) of 0.1.\
- **(f) New Output Calculation**: Computed new output values (\uc0\u55349 \u56475  and \u55349 \u56450 ) with updated weights.\
- **(g) New SSE Calculation**: Calculated the SSE with the new weights.\
- **(h) Error Reduction Analysis**: Analyzed the reduction in error achieved by the network with new weights compared to the original weights.\
\
## Part II: Neural Network from Scratch\
\
**2. Neural Network Implementation in Python**\
\
**Objective**: Developing a program to train the neural network based on the calculations from Part I.\
\
**Implementation**:\
- **(1) Program Code**: Wrote Python code to replicate the network architecture and calculations.\
- **(2) Execution Results**: Ensured that the outputs from the program matched the manual calculations.\
- **(3) Output Comparison**: Compared program execution results with manual calculations for consistency.\
\
**Additional Task**:\
- **(i) SSE Plot**: Generated a plot showing the SSE of the network changing during training.\
\
## Logistic Activation Neural Network\
\
**3. Implementing Sigmoid Activation Function**\
\
**Objective**: Modifying the neural network to use the logistic (sigmoid) activation function.\
\
**Tasks**:\
- Developed the program using sigmoid activation and randomly sampled initial weights.\
- Generated a plot showing the SSE changes during training over 100 epochs.\
\
## Part III: Wine Quality Prediction\
\
**4. Neural Network Application in Predicting Wine Quality**\
\
**Objective**: Creating a model to predict wine quality using a dataset.\
\
**Tasks**:\
- **Data Preparation**: Loaded and standardized the red wine dataset.\
- **Model Design**: Designed a neural network with specific architecture and layers.\
- **Model Training**: Configured the model with MSE as the loss function, used "Adam" for optimization, and trained for 25 epochs.\
- **Model Evaluation**: Presented performance metrics (MSE, MAE, R2) on the validation set.\
- **Visualization**: Plotted loss curves and a scatter plot for actual vs. predicted quality scores.\
\
\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 ## Part IV: Comparison of Network Models\
\
**Objective**: To develop two distinct neural network models with varying numbers of hidden layers and compare their performance using the "Wine Quality" dataset.\
\
### Data Preparation\
- Loaded the red wine dataset (`winequality-red.csv`).\
- Standardized features by centering on the mean and scaling to unit variance.\
- Split data into an 80% training set and a 20% testing set.\
\
### Model Design\
**NetworkTwoHidden**:\
- Fully connected neural network with three layers.\
- Input layer with 11 neurons for descriptive variables.\
- Two hidden layers: first with 11 neurons, second with 8 neurons.\
- Output layer with a single neuron.\
- ReLU activation for all layers except the output layer, which uses a linear activation function.\
\
**NetworkOneHidden**:\
- Fully connected neural network with two layers.\
- Input layer with 11 neurons for descriptive variables.\
- One hidden layer with 11 neurons.\
- Output layer with a single neuron.\
- ReLU activation for all layers except the output layer, which uses a linear activation function.\
\
### Model Training\
- Configured both models for training with MSE as the loss function.\
- Used "Adam" for optimization.\
- Trained models for a fixed number of epochs, with a batch size of 10 examples per gradient update.\
- Set the learning rate to 0.001.\
\
### Model Comparison\
- **(f) Loss Curves Plot**: Created plots illustrating the loss curves for both models to visualize the change in training loss values over epochs.\
- **(g) MSE Curves Plot**: Generated plots to display the MSE curves for both models, facilitating a comparison of the change in MSE values over epochs.\
\
### Analysis\
- **(h) Performance Analysis**: Analyzed and compared the performance of both models based on the plots. Discussed findings concerning network depth, validation loss, and observed trends or differences in model learning.\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 \
## Part V: Handwritten Digit Classification\
\
**5. Classifying Handwritten Digits with Neural Networks**\
\
**Objective**: Building a classifier for the MNIST dataset.\
\
**Tasks**:\
- **Data Preparation**: Loaded the MNIST dataset and prepared it for the neural network.\
- **Model Design**: Created a network with one hidden layer and softmax output layer.\
- **Model Training**: Configured and trained the model, setting aside validation data.\
- **Model Evaluation**: Evaluated model performance on a test set and reported various metrics.\
- **Visualization**: Displayed training/validation loss curves and sample classified images.\
\
## Part VI: Network Overfitting Handling\
\
**6. Addressing Overfitting in Neural Networks**\
\
**Objective**: Developing a neural network with dropout to address overfitting.\
\
**Tasks**:\
- **Network Configuration**: Added a dropout layer to the architecture from Part V.\
- **Evaluation and Visualization**: Created a plot for training and validation loss curves.\
- **Analysis**: Compared loss curves from Part V, observing the effectiveness of dropout layers.\
\
## Conclusion\
\
This project has been an incredible journey of learning, discovery, and practical application. From theoretical foundations to advanced applications, each part of this project pushed the boundaries of my understanding and skills in neural networks.\
\
## Repository Structure\
\
Each part of this project is meticulously documented, with detailed explanations and code annotations for clarity. This README serves as a comprehensive guide through my exploration. Please use the src folder to find source code to each part of this project}