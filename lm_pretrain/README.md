# Pretrained bidirectional language models for protein secondary structure prediction
This package contains the scripts for training and evaluating bidirectional language
models, as well as incorporating such pretrained bd-LMs into bd-RNNs for protein
secondary structure prediction.

## File Overview
- `__main__.py`: Command line interface
- `hparams.py`: Hyperparameters for models
- `model.py`: Contains a model class for both pretrained LMs and a BDRNN
- `pretrain.py`: Do pretraining or training of a model
- `evaluate.py`: Evaluate a trained model
- `dataset.py`: Create input data pipelines using tf.data

## Usage
Both training and evaluation expect datasets in the same format as the CPDB dataset
produced via the `make_tfrecords.py` script in the `pssp_lstm` package.

In the examples below, all bash commands are executed from the parent directory of
this package.

### Training a language model
The main difference between this package and the `pssp_lstm` package is the 
integration of pretrained language models into a BDRNN. The first step to doing
this is therefore the pretraining itself.

The command line interface allows you to train forward and backward unidirectional
language models separately, as specified by a flag. Eventually, it will be 
possible for the architecture of the LMs to be arbitrary, but for the moment they
are forced to be single-layer LSTM RNNs with the same number of units as each
other, as well as with the eventual BDRNN with which they will be combined.

To train a forwards language model on the CPDB dataset with logging, execute:
```bash
python -m lm_pretrain train /path/to/data /path/to/logdir --model lm --lm_kind fw --logging
```
Training a backwards LM consists of setting `--lm_kind` to `bw`.

### Training a BDRNN incorporating pretrained LMs
With both a forwards and backwards LM trained, they can be loaded as the first
layer of a BDRNN for further fine-tuning. 

The command for doing this is
```bash
python -m lm_pretrain train /path/to/data /path/to/logdir --model bdrnn --lm_fw_ckpt /path/to/fw/lm/ckpt/ckpt-XXXX --lm_bw_ckpt /path/to/bw/lm/ckpt/ckpt-YYYY --logging
```
When loading pretrained LMs from checkpoints, a specific step is chosen to load from. 
Which step this is depends on how the training went for a particular LM, and is 
likely to be different for the forwards and backwards models.

#### Fixing the language model layers
By default, the above command will also train the pretrained layers during 
fine-tuning. This is sometimes undesireable, so to freeze those layers,
the `--fixed_lm` flag can be passed in. This will stop the gradient 
from flowing back into the language models during training. 

## Comments
This package is still a work in progress. Eventually, there will be much more
freedom of choice of architecture for both the language models and the bdrnn,
as well as how they are combined. 
