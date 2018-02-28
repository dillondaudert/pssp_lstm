
# Predicting Secondary Structure Using Long Short-Term Memory

The purpose of this repo is to implement the model presented [here](https://arxiv.org/pdf/1412.7828.pdf) in order to reproduce the results claimed in that paper.
### Abstract
> Prediction of protein secondary structure from the amino acid sequence is a classical bioinformatics problem. Common methods use feed forward neural networks or SVM’s combined with a sliding window, as these models
does not naturally handle sequential data. Recurrent neural networks are an generalization of the feed forward
neural network that naturally handle sequential data. We use a bidirectional recurrent neural network with long
short term memory cells for prediction of secondary structure and evaluate using the CB513 dataset. On the
secondary structure 8-class problem we report better performance (0.674) than state of the art (0.664). Our model
includes feed forward networks between the long short term memory cells, a path that can be further explored.

# High-Level Overview

- **model.py**: Bidirectional LSTM RNN class
- **dataset.py**: Data input pipeline
- **hparams.py**: Specify hyperparameters for the model
- **metrics.py**: Custom streaming confusion matrix 
- **train.py**: Train a model
- **evaluate.py**: Evaluate a model
- **pssp_lstm.py**: Driver/command line tool

# User Guide

## Requirements

- Python 3
- Numpy
- Pandas
- [TensorFlow](https://www.tensorflow.org/install/) >= v1.5


## Downloading and Preparing the Data

The dataset used in this paper can be downloaded [here](https://www.princeton.edu/~jzthree/datasets/ICML2014/). The input pipeline uses the TensorFlow Dataset API, so the data files need to be converted into TFRecords. Assuming you've downloaded the .npy.gz files to the directory *datadir*, you can create the .tfrecords files by invoking:
```
./make_tfrecords -d datadir
```
This will create the training, validation, and test datasets. See [make_tfrecords.py](./make_tfrecords.py) for more details and [MakeTFRecords.ipynb](./MakeTFRecords.ipynb) for a walkthrough on what the script does.

**NOTE**: The numpy data and tfrecords data will use about 4GB of space combined.

## Training a Model

Assuming the TF records datasets have been created, the quickest way to train a model is by invoking
```
./pssp_lstm.py train /path/to/data/dir /path/to/log/dir
```
This will train the model as described on the CullPDB dataset. By default, this will train for 125 epochs with a batch size of 128. During training, you should see regular status messages: 
```
...
Step: 75, Training Loss: 1.687496, Avg Step/Sec: 2.27
Step: 90, Training Loss: 1.663782, Avg Step/Sec: 2.09
Step: 100, Eval Loss: 1.637190, Eval Accuracy: 0.331677
...
```

Depending on your hardware, you may want to use different batch sizes. The model and training hyperparameters can be adjusted in the [hparams.py](./hparams.py) file.

## Visualizing Training

If the `--logging` flag is passed during training, then summaries will be saved for a number of training statistics, including training loss, validation loss, accuracy, confusion matrices, weight histograms and distributions, etc.

These can be visualized using [Tensorboard](https://github.com/tensorflow/tensorboard). If the logs are saved to `logdir`, you can start the tensorboard server by calling
```
tensorboard --logdir logdir
```
By default, this will start a web server on the local host. 

**NOTE**: These summaries can use a lot of space, on the order of 5 - 10GB.

## Evaluating the Model

Once a model is trained, you can evaluate it vs. the CullPDB 513 test set with the following call:
```
./pssp_lstm.py evaluate /path/to/data/dir /path/to/ckpts/ckpt
```
Here, the second argument is not a directory, but instead the path to a model checkpoint previously saved during training. These will be under `logdir/ckpt/`, where `logdir` was the directory specified at training, and have names of the form `ckpt-3000*`. There will be multiple files that begin this way per checkpoint, and only the prefix is necessary. One example of evaluating a model might look like
```
./pssp_lstm.py evaluate /home/user/data/ /home/user/models/ckpt/ckpt-3000
```

# Resources


- The dataset uses proteins from the [Protein Databank](https://www.wwpdb.org/)
- Secondary structure labels are assigned according to the Dictionary of Protein Secondary Structure [(DSSP)](http://swift.cmbi.ru.nl/gv/dssp/) labels

# Additional References

1. Sixty-five years of the long march in protein secondary structure prediction: the final stretch? Yang, Y., *et al.* [Link](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbw129/2769436)
