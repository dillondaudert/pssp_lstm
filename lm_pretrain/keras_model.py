# A LM / BDRNN model in Keras

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, LSTMCell, RNN, Dense, concatenate, Masking
from tensorflow.keras.models import Model
from .dataset import create_dataset

def bdrnn_model(hparams):

    # create a bidirectional language model
    seq_id = Input(shape=tf.TensorShape([]))
    seq_lens = Input(shape=tf.TensorShape([]))
    seq_in = Input(shape=(None, 23))
    phyche_in = Input(shape=(None, hparams.num_phyche_features))
    pssm_in = Input(shape=(None, 21))

    # mask out zeros
    seq_in_mask = Masking(mask_value=0.)(seq_in)
    phyche_in_mask = Masking(mask_value=0.)(phyche_in)
    pssm_in_mask = Masking(mask_value=0.)(pssm_in)

    # create the lm
    # TODO: the LM inputs need to have SOS and EOS pre/appended
    embed = Dense(units=10, use_bias=False, name="embed")(seq_in_mask)
    lm_x = concatenate([embed, phyche_in_mask])

    fw_cells = [LSTMCell(units=hparams.lm_num_units) for i in range(hparams.lm_num_layers)]
    bw_cells = [LSTMCell(units=hparams.lm_num_units) for i in range(hparams.lm_num_layers)]

    lm_fw_out = RNN(fw_cells, return_sequences=True, name="lm_fw")(lm_x)
    lm_bw_out = RNN(bw_cells, return_sequences=True, name="lm_bw", go_backwards=True)(lm_x)
    lm_rnn_out = concatenate([lm_fw_out, lm_bw_out])

    lm_dense1 = Dense(units=200, activation="relu", name="lm_dense1")(lm_rnn_out)
    lm_dense2 = Dense(units=100, activation="relu", name="lm_dense2")(lm_dense1)

    # the language model output
    lm_out = Dense(units=23, activation="softmax", name="lm_out")(lm_dense2)

    # --------

    # create inputs to bdrnn
    x = concatenate([embed, phyche_in_mask, pssm_in_mask, lm_dense2])

    fw_cells = [LSTMCell(units=hparams.num_units) for _ in range(hparams.num_layers)]
    bw_cells = [LSTMCell(units=hparams.num_units) for _ in range(hparams.num_layers)]

    fw_out = RNN(fw_cells, return_sequences=True)(x)
    bw_out = RNN(bw_cells, return_sequences=True, go_backwards=True)(x)
    rnn_out = concatenate([fw_out, bw_out])

    dense1 = Dense(units=100, activation="relu")(rnn_out)
    dense2 = Dense(units=50, activation="relu")(dense1)

    out = Dense(units=10, activation="softmax")(dense2)

    return Model(inputs=[seq_id, seq_lens, seq_in, phyche_in, pssm_in],
                 outputs=[out])



if __name__ == "__main__":
    hparams = tf.contrib.training.HParams(
        num_phyche_features=7,
        train_file="/home/dillon/data/cpdb2/cpdb_train.tfrecords",
        valid_file="/home/dillon/data/cpdb2/cpdb_valid.tfrecords",
        test_file="/home/dillon/data/cpdb2/cpdb513_test.tfrecords",
        num_layers=2,
        num_units=300,
        batch_size=64,
        num_epochs=10,
        steps_per_epoch=100,
        validation_steps=10,
        model="bdrnn",
        )
    lm_hparams = tf.contrib.training.HParams(
        num_phyche_features=7,
        num_layers=1,
        num_units=400,
        )
    # set session
    sess = tf.Session()
    tf.keras.backend.set_session(sess)

    train_dataset = create_dataset(hparams, tf.contrib.learn.ModeKeys.TRAIN)
    valid_dataset = create_dataset(hparams, tf.contrib.learn.ModeKeys.EVAL)
    test_dataset = create_dataset(hparams, tf.contrib.learn.ModeKeys.EVAL)

    # initialize tables
    sess.run([tf.tables_initializer()])

    model = bdrnn_model(hparams, lm_hparams)
    model.summary()

    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(x=train_dataset,
              epochs=hparams.num_epochs,
              steps_per_epoch=hparams.steps_per_epoch,
              validation_data=valid_dataset,
              validation_steps=hparams.validation_steps)

    print(model.predict(x=test_dataset, steps=1))


