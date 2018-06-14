# A LM / BDRNN model in Keras

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, LSTMCell, RNN, Dense, concatenate, Masking
from tensorflow.keras.models import Model
from .dataset import create_dataset

def lm_model(hparams):

    # create a bidirectional language model
    seq_id = Input(shape=tf.TensorShape([]))
    seq_lens = Input(shape=tf.TensorShape([]))
    seq_in = Input(shape=(None, 23))
    phyche_in = Input(shape=(None, hparams.num_phyche_features))
    pssm_in = Input(shape=tf.TensorShape([])) # there are no profiles for the LM

    seq_in_mask = Masking(mask_value=0.)(seq_in)
    phyche_in_mask = Masking(mask_value=0.)(phyche_in)

    # amino acid embeddings
    embed = Dense(units=10, use_bias=False)(seq_in_mask)

    # concatenate embeddings and phyche
    x = concatenate([embed, phyche_in_mask])

    fw_cells = [LSTMCell(units=hparams.num_units) for _ in range(hparams.num_layers)]
    bw_cells = [LSTMCell(units=hparams.num_units) for _ in range(hparams.num_layers)]

    fw_out = RNN(fw_cells, return_sequences=True)(x)
    bw_out = RNN(bw_cells, return_sequences=True, go_backwards=True)(x)
    rnn_out = concatenate([fw_out, bw_out])

    dense1 = Dense(units=200, activation="relu")(rnn_out)
    dense2 = Dense(units=100, activation="relu")(dense1)

    lm_out = Dense(units=23, activation="softmax")(dense2)

    model = Model(inputs=[seq_id, seq_lens, seq_in, phyche_in, pssm_in], outputs=[lm_out])
    return model

if __name__ == "__main__":
    hparams = tf.contrib.training.HParams(
        num_phyche_features=7,
        train_file="/home/dillon/data/cUR50/cUR50_train.tfrecords",
        valid_file="/home/dillon/data/cUR50/cUR50_valid.tfrecords",
        num_layers=2,
        num_units=100,
        batch_size=32,
        num_epochs=2,
        steps_per_epoch=50,
        validation_steps=25,
        model="lm")
    # set session
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    
    train_dataset = create_dataset(hparams, tf.contrib.learn.ModeKeys.TRAIN)
    valid_dataset = create_dataset(hparams, tf.contrib.learn.ModeKeys.EVAL)

    # initialize tables
    sess.run([tf.tables_initializer()])
    
    model = lm_model(hparams)
    model.summary()

    


    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(x=train_dataset,
              epochs=hparams.num_epochs,
              steps_per_epoch=hparams.steps_per_epoch,
              validation_data=valid_dataset,
              validation_steps=hparams.validation_steps)
