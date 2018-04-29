#!/usr/bin/env python3
"""Driver for training and running models."""
import argparse as ap
from pathlib import Path
from .train import train
from .evaluate import evaluate
from .hparams import HPARAMS

def main():


    # Define the main argument parser
    parser = ap.ArgumentParser(prog="pssp_lstm", description="Train and run models.",
                               argument_default=ap.SUPPRESS)

    subparsers = parser.add_subparsers(title='subcommands')

    # -- training subparser --
    tr_parser = subparsers.add_parser("train", help="Train a model")

    tr_parser.add_argument("datadir", type=str,
                           help="the directory where the .tfrecords data is located")

    tr_parser.add_argument("logdir", type=str,
                           help="the directory where model checkpoints and logs will\
                                 be saved")
    tr_parser.add_argument("-l", "--logging", action="store_true",
                           help="toggle to enable tf.summary logs (disabled by default)")

    tr_parser.set_defaults(entry="train")

    ev_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")

    ev_parser.add_argument("datadir", type=str,
                           help="the directory where the cpdb_513.tfrecords file is located.")
    ev_parser.add_argument("ckpt", type=str,
                           help="a checkpoint file for a trained model. This will\
                                 look something like this:\
                                 /path/ckpt/ckpt-4100")
    ev_parser.set_defaults(entry="evaluate")

    args = parser.parse_args()


    if args.entry == "train":
        # run training
        HPARAMS.logging = args.logging

        logpath = Path(args.logdir)
        HPARAMS.logdir = str(logpath.absolute())
        HPARAMS.train_file = str(Path(args.datadir, "cpdb_train.tfrecords").absolute())
        HPARAMS.valid_file = str(Path(args.datadir, "cpdb_valid.tfrecords").absolute())

        train(HPARAMS)

    elif args.entry == "evaluate":
        # evaluate a trained network on the test data
        HPARAMS.valid_file = str(Path(args.datadir, "cpdb_513.tfrecords").absolute())
        HPARAMS.model_ckpt = str(Path(args.ckpt).absolute())

        evaluate(HPARAMS)



if __name__ == "__main__":
    main()
