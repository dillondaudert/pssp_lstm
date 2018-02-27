#!/usr/bin/env python3
# driver and command line
import argparse as ap
from pathlib import Path
#from train import train
from hparams import HPARAMS

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

    tr_parser.set_defaults(entry="train")

    args = parser.parse_args()

    if args.entry == "train":
        # run training

        logpath = Path(args.logdir)
        HPARAMS.train_file = str(Path(args.datadir, "cpdb_train.tfrecords").absolute())
        HPARAMS.valid_file = str(Path(args.datadir, "cpdb_valid.tfrecords").absolute())
        HPARAMS.test_file = str(Path(args.datadir, "cpdb_513_test.tfrecords").absolute())

        print(HPARAMS)
        quit()

#        train(HPARAMS)

if __name__ == "__main__":
    main()
