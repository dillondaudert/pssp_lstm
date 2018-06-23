# A LM / BDRNN model in Keras

import time

from comet_ml import Experiment

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, GRUCell, LSTMCell, RNN, Dense, concatenate, Masking, Bidirectional, Lambda
from tensorflow.keras.models import Model
from .dataset import create_dataset
from .lookup import create_lookup_table

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

    if hparams.rnn_cell == "gru":
        RNNCell = GRUCell
    else:
        RNNCell = LSTMCell

    lm_cells = [RNNCell(units=hparams.lm_num_units,
                        dropout=hparams.lm_dropout,
                        recurrent_dropout=hparams.lm_dropout,
                        implementation=2) for _ in range(hparams.lm_num_layers)]

    lm_rnn_out = Bidirectional(RNN(lm_cells, return_sequences=True, name="lm_rnn"), merge_mode="concat")(lm_x)

    lm_dense1 = Dense(units=200, activation="relu", name="lm_dense1")(lm_rnn_out)
    lm_dense2 = Dense(units=100, activation="relu", name="lm_dense2")(lm_dense1)

    lm_embed = Dense(units=10, name="lm_embed")(lm_dense2)

    # the language model output
    lm_out = Dense(units=23, activation="softmax", name="lm_out")(lm_dense2)

    # --------

    # create inputs to bdrnn
    x = concatenate([embed, lm_embed, pssm_in_mask])

    cells = [RNNCell(units=hparams.num_units,
                     dropout=hparams.dropout,
                     recurrent_dropout=hparams.dropout,
                     implementation=2) for _ in range(hparams.num_layers)]

    rnn_out = Bidirectional(RNN(cells, return_sequences=True, name="bdrnn_rnn"), merge_mode="concat")(x)

    dense1 = Dense(units=100, activation="relu")(rnn_out)
    dense2 = Dense(units=50, activation="relu")(dense1)

    out = Dense(units=10, activation="softmax")(dense2)
#    pred_indices = Lambda(lambda x: keras.backend.argmax(x))(out)
#    preds = Lambda(lambda x: keras.backend.one_hot(x, 10))(pred_indices)

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
        rnn_cell="gru",
        num_layers=2,
        num_units=300,
        dropout=0.,
        lm_num_layers=2,
        lm_num_units=300,
        lm_dropout=0.,
        batch_size=50,
        num_epochs=40,
        steps_per_epoch=75,
        validation_steps=10,
        model="bdrnn",
        )
    experiment = Experiment(api_key="",
                            project_name="thesis-baseline",
                            auto_param_logging=False)
    experiment.log_multiple_params(hparams.values())


    g = tf.Graph()
    with tf.Session(graph=g) as sess:

        # create lookup tables
        hparams.prot_lookup_table = create_lookup_table("prot")
        hparams.prot_reverse_lookup_table = create_lookup_table("prot", reverse=True)
        hparams.struct_lookup_table = create_lookup_table("struct")
        hparams.struct_reverse_lookup_table = create_lookup_table("struct", reverse=True)

        train_dataset = create_dataset(hparams, tf.contrib.learn.ModeKeys.TRAIN)
        valid_dataset = create_dataset(hparams, tf.contrib.learn.ModeKeys.EVAL)
        test_dataset = create_dataset(hparams, tf.contrib.learn.ModeKeys.EVAL)

        # initialize tables
        sess.run([tf.tables_initializer()])#, train_iterator.initializer, valid_iterator.initializer, test_iterator.initializer])
        keras.backend.set_session(sess)

        models = create_models(hparams)
        model = models["bdrnn"]
        model.summary()

        #rmsprop = keras.optimizers.RMSprop(lr=0.001, clipnorm=0.5)
        model.compile(optimizer="rmsprop",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"],)

        #modeldir = "/home/dillon/models/thesis/baseline/%s" % (time.asctime(time.localtime()))
        modeldir = "/home/dillon/models/thesis/baseline/lm-%d-%d-%.2f%%_bdrnn-%d-%d-%.2f%%/" % (hparams.lm_num_layers,
                                                                                  hparams.lm_num_units,
                                                                                  hparams.lm_dropout,
                                                                                  hparams.num_layers,
                                                                                  hparams.num_units,
                                                                                  hparams.dropout)

        with experiment.train():
            model.fit(x=train_dataset,
                      epochs=hparams.num_epochs,
                      steps_per_epoch=hparams.steps_per_epoch,
                      validation_data=valid_dataset,
                      validation_steps=hparams.validation_steps,
                      callbacks=[keras.callbacks.EarlyStopping("val_loss", 0.001, 5, verbose=1),
                                 keras.callbacks.TensorBoard(modeldir+"/logs/"),
                                 keras.callbacks.ModelCheckpoint(modeldir+"/test_baseline.hdf5",
                                                                 monitor="val_loss",
                                                                 save_best_only=True)])

        with experiment.test():
            test_loss, test_acc = model.evaluate(x=test_dataset, steps=10)
            metrics = {
                "loss": test_loss,
                "accuracy": test_acc,
            }
            experiment.log_multiple_metrics(metrics)




