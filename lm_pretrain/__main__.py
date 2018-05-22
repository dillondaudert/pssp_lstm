#!/usr/bin/env python3
"""Driver for training and running models."""
import argparse as ap
from pathlib import Path
from .pretrain import pretrain
from .hparams import hparams

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
    parser.add_argument("-m", "--model", type=str, choices=["lm", "bdrnn"], required=True,
                           help="which kind of model to train")
    parser.add_argument("--large", action="store_true", default=False,
                           help="toggle whether to use the large version of the model")
    parser.add_argument("--lm_kind", type=str, choices=["fw", "bw"], default="fw",
                           help="whether to train a forward or backward language model\
                                   (default: forward)")
    parser.add_argument("--lm_fw_ckpt", type=str, default="",
                           help="the path to a pretrained forward language model checkpoint")
    parser.add_argument("--lm_bw_ckpt", type=str, default="",
                           help="the path to a pretrained backward language model checkpoint")

    args = parser.parse_args()

    model_hparams = args.model if not args.large else args.model+"_large"
    HPARAMS = hparams[model_hparams]
    HPARAMS.lm_fw_ckpt = args.lm_fw_ckpt
    HPARAMS.lm_bw_ckpt = args.lm_bw_ckpt
    print("Model: %s, HPARAMS: %s" % (args.model, model_hparams))

    if args.model == "bdrnn":
        if (args.lm_fw_ckpt != "" and args.lm_bw_ckpt == "") or (args.lm_fw_ckpt == "" and args.lm_bw_ckpt != ""):
            print("Both lm_fw_ckpt and lm_bw_ckpt must either be paths or left out. Quitting.")
            quit()
        elif args.lm_fw_ckpt != "":
            HPARAMS.pretrained = True
    else:
        HPARAMS.lm_kind = args.lm_kind

    # run training
    HPARAMS.logging = args.logging

    logpath = Path(args.logdir)
    HPARAMS.logdir = str(logpath.absolute())
    HPARAMS.train_file = str(Path(args.datadir, "cpdb_train.tfrecords").absolute())
    HPARAMS.valid_file = str(Path(args.datadir, "cpdb_valid.tfrecords").absolute())


    pretrain(HPARAMS)


if __name__ == "__main__":
    main()
