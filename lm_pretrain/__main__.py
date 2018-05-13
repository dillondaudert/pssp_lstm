#!/usr/bin/env python3
"""Driver for training and running models."""
import argparse as ap
from pathlib import Path
from .pretrain import pretrain
from .hparams import HPARAMS

def main():

    # Define the main argument parser
    parser = ap.ArgumentParser(prog="lm_pretrain", description="Pretrain a model",
                               argument_default=ap.SUPPRESS)

    # -- training subparser --
    parser.add_argument("datadir", type=str,
                           help="the directory where the .tfrecords data is located")

    parser.add_argument("logdir", type=str,
                           help="the directory where model checkpoints and logs will\
                                 be saved")
    parser.add_argument("-l", "--logging", action="store_true",
                           help="toggle to enable tf.summary logs (disabled by default)")

    args = parser.parse_args()

    # run training
    HPARAMS.logging = args.logging

    logpath = Path(args.logdir)
    HPARAMS.logdir = str(logpath.absolute())
    HPARAMS.train_file = str(Path(args.datadir, "cpdb_train.tfrecords").absolute())
    HPARAMS.valid_file = str(Path(args.datadir, "cpdb_valid.tfrecords").absolute())

    train(HPARAMS)


if __name__ == "__main__":
    main()
