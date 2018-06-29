#!/usr/bin/env python3
"""Driver for training and running models."""
import argparse as ap
from pathlib import Path
from .pretrain import pretrain
from .evaluate import evaluate
from .hparams import hparams

def main():

    # Define the main argument parser
    parser = ap.ArgumentParser(prog="lm_pretrain", description="Pretrain a model",
                               argument_default=ap.SUPPRESS)

    subparsers = parser.add_subparsers(title="subcommands")

    # -- training subparser --
    tr_parser = subparsers.add_parser("train", help="Train a model")

    tr_parser.add_argument("datadir", type=str,
                           help="the directory where the .tfrecords data is located")

    tr_parser.add_argument("logdir", type=str,
                           help="the directory where model checkpoints and logs will\
                                 be saved")
    tr_parser.add_argument("-l", "--logging", action="store_true",
                           help="toggle to enable tf.summary logs (disabled by default)")
    tr_parser.add_argument("-m", "--model", type=str, choices=["bdlm", "bdrnn"], required=True,
                           help="which kind of model to train")
    tr_parser.add_argument("--bdlm_ckpt", type=str, default="",
                           help="the path to a pretrained language model checkpoint")
    tr_parser.add_argument("--fixed_lm", action="store_true",
                           help="this flag indicates that the pretrained models should be\
                                 fixed during fine-tuning.")
    tr_parser.set_defaults(entry="train")

    ev_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")

    ev_parser.add_argument("datadir", type=str,
                           help="the directory where the cpdb_513.tfrecords file is located")
    ev_parser.add_argument("ckpt", type=str, help="a tf model checkpoint file.")

    ev_parser.add_argument("-m", "--model", type=str, choices=["bdlm", "bdrnn"], required=True,
                           help="which kind of model to train")

    ev_parser.set_defaults(entry="evaluate")

    args = parser.parse_args()

    if args.entry == "train":
        model_hparams = args.model
        HPARAMS = hparams[model_hparams]
        print("Model: %s, HPARAMS: %s" % (args.model, model_hparams))

        if args.model == "bdrnn":
            if args.bdlm_ckpt != "":
                HPARAMS.pretrained = True

        # run training
        HPARAMS.logging = args.logging

        logpath = Path(args.logdir)
        HPARAMS.logdir = str(logpath.absolute())
        HPARAMS.train_file = str(Path(args.datadir, HPARAMS.train_file).absolute())
        HPARAMS.valid_file = str(Path(args.datadir, HPARAMS.valid_file).absolute())
        HPARAMS.test_file = str(Path(args.datadir, HPARAMS.test_file).absolute())

        pretrain(HPARAMS)

    elif args.entry == "evaluate":
        model_hparams = args.model if not args.large else args.model+"_large"
        HPARAMS = hparams[model_hparams]
        HPARAMS.valid_file = str(Path(args.datadir, "cpdb_513.tfrecords").absolute())
        HPARAMS.model_ckpt = str(Path(args.ckpt).absolute())
        HPARAMS.pretrained = True

        evaluate(HPARAMS)


    else:
        print("Unrecognized command, exiting.")


if __name__ == "__main__":
    main()
