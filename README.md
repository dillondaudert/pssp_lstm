# Predicting Secondary Structure with Deep Convolutional and Recurrent Neural Networks
This repo contains implementations of several recent papers using deep learning to predict protein secondary structure. Each folder contains a README with further details.

## [Protein Secondary Structure Prediction with Long Short Term Memory Networks. Sonderby, S., & Winther, O. 2015](https://arxiv.org/pdf/1412.7828.pdf)
Folder: [pssp_lstm](./pssp_lstm/)

**Status**: Implemented
### Abstract
> Prediction of protein secondary structure from the amino acid sequence is a classical bioinformatics problem. Common methods use feed forward neural networks or SVM’s combined with a sliding window, as these models
does not naturally handle sequential data. Recurrent neural networks are an generalization of the feed forward
neural network that naturally handle sequential data. We use a bidirectional recurrent neural network with long
short term memory cells for prediction of secondary structure and evaluate using the CB513 dataset. On the
secondary structure 8-class problem we report better performance (0.674) than state of the art (0.664). Our model
includes feed forward networks between the long short term memory cells, a path that can be further explored.

## [Protein Secondary Structure Prediction Using Cascaded Convolutional and Recurrent Neural Networks. Li, Z., & Yu, Y. 2016](https://arxiv.org/abs/1604.07176)

**Status**: Not implemented
### Abstract
> Protein secondary structure prediction is an important problem in bioinformatics. Inspired by the recent successes of deep neural networks, in this paper, we propose an end-to-end deep network that predicts protein secondary structures from integrated local and global contextual features. Our deep architecture leverages convolutional neural networks with different kernel sizes to extract multiscale local contextual features. In addition, considering long-range dependencies existing in amino acid sequences, we set up a bidirectional neural network consisting of gated recurrent unit to capture global contextual features. Furthermore, multi-task learning is utilized to predict secondary structure labels and amino-acid solvent accessibility simultaneously. Our proposed deep network demonstrates its effectiveness by achieving state-of-the-art performance, i.e., 69.7% Q8 accuracy on the public benchmark CB513, 76.9% Q8 accuracy on CASP10 and 73.1% Q8 accuracy on CASP11. Our model and results are publicly available.

## [Capturing non-local interactions by long short-term memory bidirectional recurrent neural networks for improving prediction of protein secondary structure, backbone angles, contact numbers and solvent accessibility. Heffernan, R., Yang, Y., Paliwal, K., & Zhou, Y. *Bioinformatics* 2017](https://www.ncbi.nlm.nih.gov/pubmed/28430949)

**Status**: Not implemented
### Abstract
> The accuracy of predicting protein local and global structural properties such as secondary structure and solvent accessible surface area has been stagnant for many years because of the challenge of accounting for non-local interactions between amino acid residues that are close in three-dimensional structural space but far from each other in their sequence positions. All existing machine-learning techniques relied on a sliding window of 10-20 amino acid residues to capture some 'short to intermediate' non-local interactions. Here, we employed Long Short-Term Memory (LSTM) Bidirectional Recurrent Neural Networks (BRNNs) which are capable of capturing long range interactions without using a window.

## [Next-Step Conditioned Deep Convolutional Neural Networks Improve Protein Secondary Structure Prediction. Busia, A., & Jaitly, N. *ISMB* 2017](https://research.google.com/pubs/pub46131.html)

**Status**: Not implemented
### Abstract
> Recently developed deep learning techniques have significantly improved the accuracy of various speech and image recognition systems. In this paper we show how to adapt some of these techniques to create a novel chained convolutional architecture with next-step conditioning for improving performance on protein sequence prediction problems. We explore its value by demonstrating its ability to improve performance on eight-class secondary structure prediction. We first establish a state-of-the-art baseline by adapting recent advances in convolutional neural networks which were developed for vision tasks. This model achieves 70.0% per amino acid accuracy on the CB513 benchmark dataset without use of standard performance-boosting techniques such as ensembling or multitask learning. We then improve upon this state-of-the-art result using a novel chained prediction approach which frames the secondary structure prediction as a next-step prediction problem. This sequential model achieves 70.3% Q8 accuracy on CB513 with a single model; an ensemble of these models produces 71.4% Q8 accuracy on the same test set, improving upon the previous overall state of the art for the eight-class secondary structure problem

# User Guide

## Requirements

- Python 3
- [TensorFlow](https://www.tensorflow.org/install/) >= v1.7


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
./pssp_lstm/pssp_lstm.py train /path/to/data/dir /path/to/log/dir
```
This will train the model as described on the CullPDB dataset. By default, this will train for 125 epochs with a batch size of 128. During training, you should see regular status messages: 
```
...
Step: 75, Training Loss: 1.687496, Avg Step/Sec: 2.27
Step: 90, Training Loss: 1.663782, Avg Step/Sec: 2.09
Step: 100, Eval Loss: 1.637190, Eval Accuracy: 0.331677
...
```

Depending on your hardware, you may want to use different batch sizes. The model and training hyperparameters can be adjusted in the [hparams.py](./pssp_lstm/hparams.py) file.

## Visualizing Training

If the `--logging` flag is passed during training, then summaries will be saved for a number of training statistics, including training loss, validation loss, accuracy, confusion matrices, weight histograms and distributions, etc.

These can be visualized using [Tensorboard](https://github.com/tensorflow/tensorboard). If the logs are saved to `logdir`, you can start the tensorboard server by calling
```
tensorboard --logdir logdir
```
By default, this will start a web server on the local host. 

## Evaluating the Model

Once a model is trained, you can evaluate it vs. the CullPDB 513 test set with the following call:
```
./pssp_lstm/pssp_lstm.py evaluate /path/to/data/dir /path/to/ckpts/ckpt
```
Here, the second argument is not a directory, but instead the path to a model checkpoint previously saved during training. These will be under `logdir/ckpt/`, where `logdir` was the directory specified at training, and have names of the form `ckpt-3000*`. There will be multiple files that begin this way per checkpoint, and only the prefix is necessary. One example of evaluating a model might look like
```
./pssp_lstm/pssp_lstm.py evaluate /home/user/data/ /home/user/models/ckpt/ckpt-3000
```

# Resources


- The dataset uses proteins from the [Protein Databank](https://www.wwpdb.org/)
- Secondary structure labels are assigned according to the Dictionary of Protein Secondary Structure [(DSSP)](http://swift.cmbi.ru.nl/gv/dssp/) labels

# Additional References

1. Sixty-five years of the long march in protein secondary structure prediction: the final stretch? Yang, Y., *et al.* [Link](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbw129/2769436)
