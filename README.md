# Recurrent Neural Networks for Protein Secondary Structure Prediction
This repo contains ongoing work exploring recurrent neural network models as applied to protein secondary structure prediction.

The [pssp_lstm](./pssp_lstm/) module contains an implementation of the LSTM RNN specified in [Sonderby & Winther, 2015](https://arxiv.org/pdf/1412.7828.pdf). See the README in that folder for more details and a user guide.

The [lm_pretrain](./lm_pretrain/) module allows users to train bidirectional language models that can be combined with bidirectional RNNs for protein secondary structure prediction. See the README in that folder for more details and a user guide.

## Further Work
This repo is under development. Current work is focusing on expanding the functionality of `lm_pretrain` to allow for more flexible models, and for exploring different ways of integrating pretrained LMs into BDRNNs.

Work on implementing models from other papers is currently on pause.
