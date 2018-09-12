#!/usr/bin/env python3
"""Driver for training and running models."""
import argparse as ap
from pathlib import Path
from .pretrain import pretrain
from .evaluate import evaluate
from .analysis.analyze import analyze
from .hparams import hparams
from .hparam_helpers import hparams_to_str, HPARAM_CHOICES, HPARAM_DESCS

def main():

    # Define the main argument parser
    parser = ap.ArgumentParser(prog="lm_pretrain", description="Pretrain a model",
                               argument_default=ap.SUPPRESS)

    subparsers = parser.add_subparsers(title="subcommands")

    # -- training subparser --
    tr_parser = subparsers.add_parser("train", help="Train a model")

    tr_parser.add_argument("datadir", type=str, help=HPARAM_DESCS["datadir"][1])

    tr_parser.add_argument("logdir", type=str, help=HPARAM_DESCS["logdir"][1])

    tr_parser.add_argument("-l", "--logging", action="store_true", help=HPARAM_DESCS["logging"][1])

    tr_parser.add_argument("-m", "--model", type=str, choices=HPARAM_CHOICES["model"], required=True,
                           help=HPARAM_DESCS["model"][1])

    tr_group = tr_parser.add_mutually_exclusive_group()
    tr_group.add_argument("--bdrnn_ckpt", type=str, default="",
                           help=HPARAM_DESCS["bdrnn_ckpt"][1])
    tr_group.add_argument("--bdlm_ckpt", type=str, default="",
                           help=HPARAM_DESCS["bdlm_ckpt"][1])
    tr_parser.add_argument("--freeze_bdlm", action="store_true",
                           help=HPARAM_DESCS["freeze_bdlm"][1])
    tr_parser.add_argument("--loss_weights", nargs=2, type=float,
                           help="this flag takes\
                                 2 arguments indicating the weights for the lm loss and pssp loss\
                                 respectively. \
                                 If --freeze_bdlm is specified, this option is ignored.")
    tr_parser.set_defaults(entry="train")

    ev_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")

    ev_parser.add_argument("ckpt", type=str, help="a tf model checkpoint file.")
    ev_parser.add_argument("-m", "--model", type=str, choices=HPARAM_CHOICES["model"],
                           required=True,
                           help=HPARAM_DESCS["model"][1])
    ev_parser.add_argument("-f", "--files", nargs="+", help="a list of files to evaluate.")
    ev_parser.add_argument("-o", "--outfile", type=str,
                           help="the name of the output pickle file where the results will be saved. optional")

    ev_parser.set_defaults(entry="evaluate")

    an_parser = subparsers.add_parser("analyze", help="Analyze results")
    an_parser.add_argument("-f", "--datafiles", type=str, nargs="+", required=True)
    an_parser.add_argument("-o", "--outfile", type=str, required=True)
    an_parser.set_defaults(entry="analyze")

    args = parser.parse_args()

    if args.entry == "train":
        model_hparams = args.model
        HPARAMS = hparams[model_hparams]
        print("Model: %s, HPARAMS: %s" % (args.model, model_hparams))

        if args.bdlm_ckpt != "":
            HPARAMS.bdlm_ckpt = args.bdlm_ckpt
        elif args.bdrnn_ckpt != "":
            HPARAMS.bdrnn_ckpt = args.bdrnn_ckpt

        if args.model == "bdrnn":

            HPARAMS.freeze_bdlm = args.freeze_bdlm
            if not args.freeze_bdlm and args.loss_weights is not None:
                HPARAMS.loss_weights = args.loss_weights
            LM_HPARAMS = hparams["cnn_bdlm"]
            LM_HPARAMS.freeze_bdlm = args.freeze_bdlm
            HPARAMS.lm_hparams = LM_HPARAMS

        # run training
        HPARAMS.logging = args.logging

        logpath = Path(args.logdir)
        HPARAMS.logdir = str(logpath.absolute())
        if "file_pattern" in vars(HPARAMS):
            if "num_train_files" not in vars(HPARAMS) or "num_valid_files" not in vars(HPARAMS):
                print("num_train_files and num_valid_files must both be specified if file_pattern is given.\nQuitting.")
                quit()
            HPARAMS.file_pattern = str(Path(args.datadir, HPARAMS.file_pattern))
        else:
            HPARAMS.train_file = str(Path(args.datadir, HPARAMS.train_file).absolute())
            HPARAMS.valid_file = str(Path(args.datadir, HPARAMS.valid_file).absolute())
            HPARAMS.test_file = str(Path(args.datadir, HPARAMS.test_file).absolute())

        print("Hyperparameters")
        print(hparams_to_str(HPARAMS))
        if "lm_hparams" in vars(HPARAMS):
            print("\nLanguage Model Hyperparameters")
            print(hparams_to_str(HPARAMS.lm_hparams))

        cont = input("Continue? [y]/n: ")
        if cont == "" or cont == "y":
            print("Continuing.")
            pretrain(HPARAMS)
        else:
            print("Quitting.")
            quit()

    elif args.entry == "evaluate":
        model_hparams = args.model
        HPARAMS = hparams[model_hparams]
        HPARAMS.batch_size = 1
        HPARAMS.ckpt = str(Path(args.ckpt).absolute())
        if args.model == "bdrnn":
            HPARAMS.freeze_bdlm = True
            LM_HPARAMS = hparams["cnn_bdlm"]
            LM_HPARAMS.freeze_bdlm = True
            HPARAMS.lm_hparams = LM_HPARAMS

        if args.outfile is not None:
            outfile = str(Path(args.outfile))
        else:
            outfile = None

        evaluate(HPARAMS, args.files, outfile)

    elif args.entry == "analyze":
        analyze(args.datafiles, args.outfile)

    else:
        print("Unrecognized command, exiting.")


if __name__ == "__main__":
    main()
