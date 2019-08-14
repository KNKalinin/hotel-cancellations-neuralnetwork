# Predicting Hotel Cancellations with a Keras Neural Network: Part 2

In a previous post, a support vector machine (SVM) was used to predict whether a customer that had made a hotel booking would ultimately cancel or not.

Through building the models on the training set and then validating them against a separate test set, the AUC score came in at **0.74**.

The purpose of this follow up article is to determine whether a neural network built using Keras demonstrates higher accuracy in predicting hotel cancellations for the test set. In other words, can a higher AUC be achieved by using a neural network instead of an SVM?

## Scaling data

The full code containing the output in a Jupyter notebook is available at the following GitHub repository.

