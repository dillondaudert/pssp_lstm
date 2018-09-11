"""Evaluate the model"""
# basic example of training a network end-to-end
from time import process_time
from pathlib import Path
import tensorflow as tf, numpy as np
import pandas as pd
from .model_helper import create_model

def evaluate(hparams, files, outfile=None):
    """Evaluate a trained model"""

    cols = ["file", "len", "id", "seq", "phyche", "pssm", "logits", "ss"]
    hcols = ["h_0", "h_1", "h_2", "lm_logits"]
    recs = []

    for f in files:
        hparams.valid_file = f
        eval_tuple = create_model(hparams, tf.contrib.learn.ModeKeys.EVAL)
        with eval_tuple.graph.as_default():
            local_initializer = tf.local_variables_initializer()

        print("Evaluating model on %s" % (hparams.valid_file))
        # do evaluation
        eval_tuple.model.saver.restore(eval_tuple.session, hparams.ckpt)
        eval_tuple.session.run([eval_tuple.iterator.initializer, local_initializer])
        while True:
            try:
                fetched = eval_tuple.model.named_eval(eval_tuple.session)

                if "filter_size" in vars(hparams):
                    k = hparams.filter_size
                else:
                    k = 1

                rec = (f,
                       fetched["inputs"].id[0],
                       fetched["inputs"].len[0],
                       fetched["inputs"].seq[0],
                       fetched["inputs"].phyche[0, k:-k, :],
                       fetched["inputs"].pssm[0],
                       fetched["logits"][0],
                       fetched["inputs"].ss[0]
                       )
                if "outputs" in fetched.keys():
                    rec = rec + (fetched["outputs"].h_0[0],
                                 fetched["outputs"].h_1[0],
                                 fetched["outputs"].h_2[0],
                                 fetched["outputs"].lm_logits[0],
                                 )

                for t in rec[3:]:
                    # assert that all input/output tensors are of same sequence length
                    assert t.shape[0] == fetched["inputs"].len[0]

                recs.append(rec)

                # summary_writer.add_summary(summary, global_step)
            except tf.errors.OutOfRangeError:
                break

    df_cols = cols if len(cols) == len(recs[0]) else cols+hcols

    df = pd.DataFrame.from_records(data=recs, columns=df_cols)
    print(df.iloc[-1])

    df = df.reindex(columns = cols+hcols)
    print(df.iloc[-1])

    if outfile is not None:
        df.to_pickle(outfile)
