
# Predicting Secondary Structure Using Long Short-Term Memory

The purpose of this repo is to implement the model presented [here](https://arxiv.org/pdf/1412.7828.pdf) in order to reproduce the results claimed in that paper.
### Abstract
> Prediction of protein secondary structure from the amino acid sequence is a classical bioinformatics problem. Common methods use feed forward neural networks or SVM’s combined with a sliding window, as these models
does not naturally handle sequential data. Recurrent neural networks are an generalization of the feed forward
neural network that naturally handle sequential data. We use a bidirectional recurrent neural network with long
short term memory cells for prediction of secondary structure and evaluate using the CB513 dataset. On the
secondary structure 8-class problem we report better performance (0.674) than state of the art (0.664). Our model
includes feed forward networks between the long short term memory cells, a path that can be further explored.

# Overview

- **model.py**: Bidirectional LSTM RNN class
- **dataset.py**: Data input pipeline
- **hparams.py**: Specify hyperparameters for the model
- **metrics.py**: Custom streaming confusion matrix 
- **train.py**: Train a model
- **evaluate.py**: Evaluate a model
- **pssp_lstm.py**: Driver/command line tool
