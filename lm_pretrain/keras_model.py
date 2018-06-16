# A LM / BDRNN model in Keras

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, LSTMCell, RNN, Dense, concatenate, Masking, Bidirectional
from tensorflow.keras.models import Model
from .dataset import create_dataset
from .lookup import create_lookup_table
import gc

def create_models(hparams):

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

    lm_cells = [LSTMCell(units=hparams.lm_num_units) for _ in range(hparams.lm_num_layers)]

    lm_rnn_out = Bidirectional(RNN(lm_cells, return_sequences=True, name="lm_rnn"), merge_mode="concat")(lm_x)

    lm_dense1 = Dense(units=200, activation="relu", name="lm_dense1")(lm_rnn_out)
    lm_dense2 = Dense(units=100, activation="relu", name="lm_dense2")(lm_dense1)

    # the language model output
    lm_out = Dense(units=23, activation="softmax", name="lm_out")(lm_dense2)

    # --------

    # create inputs to bdrnn
    x = concatenate([embed, pssm_in_mask, lm_dense2])

    cells = [LSTMCell(units=hparams.num_units) for _ in range(hparams.num_layers)]

    rnn_out = Bidirectional(RNN(cells, return_sequences=True, name="bdrnn_fw"), merge_mode="concat")(x)

    dense1 = Dense(units=100, activation="relu")(rnn_out)
    dense2 = Dense(units=50, activation="relu")(dense1)

    out = Dense(units=10, activation="softmax")(dense2)

    bdrnn_model = Model(inputs=[seq_id, seq_lens, seq_in, phyche_in, pssm_in],
                        outputs=[out])
    lm_model = Model(inputs=[seq_id, seq_lens, seq_in, phyche_in, pssm_in],
                     outputs=[lm_out])
    return {"bdrnn": bdrnn_model, "lm": lm_model}



if __name__ == "__main__":
    hparams = tf.contrib.training.HParams(
        num_phyche_features=7,
        train_file="/home/dillon/data/cpdb2/cpdb_train.tfrecords",
        valid_file="/home/dillon/data/cpdb2/cpdb_valid.tfrecords",
        test_file="/home/dillon/data/cpdb2/cpdb513_test.tfrecords",
        num_layers=1,
        num_units=200,
        batch_size=50,
        num_epochs=10,
        steps_per_epoch=60,
        validation_steps=10,
        lm_num_layers=2,
        lm_num_units=300,
        model="bdrnn",
        )
    
    # set session
    sess = tf.Session()

    # create lookup tables
    hparams.prot_lookup_table = create_lookup_table("prot")
    hparams.prot_reverse_lookup_table = create_lookup_table("prot", reverse=True)
    hparams.struct_lookup_table = create_lookup_table("struct")
    hparams.struct_reverse_lookup_table = create_lookup_table("struct", reverse=True)

    train_dataset = create_dataset(hparams, tf.contrib.learn.ModeKeys.TRAIN)
    valid_dataset = create_dataset(hparams, tf.contrib.learn.ModeKeys.EVAL)

    # initialize tables
    sess.run([tf.tables_initializer()])
    tf.keras.backend.set_session(sess)

    models = create_models(hparams)
    model = models["bdrnn"]

    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"],)
    
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="/home/dillon/models/logs/test",
                                                 histogram_freq=1,
                                                 batch_size=hparams.validation_steps)

    model.fit(x=train_dataset,
              epochs=hparams.num_epochs,
              steps_per_epoch=hparams.steps_per_epoch,
              validation_data=valid_dataset,
              validation_steps=hparams.validation_steps,
              callbacks=[tb_callback])
    
    hparams.batch_size=10
    test_dataset = create_dataset(hparams, tf.contrib.learn.ModeKeys.EVAL)
    print(model.evaluate(x=test_dataset, steps=50))


