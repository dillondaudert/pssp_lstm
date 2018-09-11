# do a grid search

import tensorflow as tf
from .bdrnn_model import BDRNNModel
from .hparam_helpers import hparams_to_str
from .hparams import hparams
from .pretrain import pretrain

def main():
    grid_params = {
            "num_layers": [1,   1,   1,   1],
            "num_units":  [256, 256, 256, 256],
            "dropout":    [.3,  .3,  .3,  .3],
            "var_dropout":[.3,  .3,  .3,  .3],
            "num_epochs": [200, 200, 200, 200],
            "sub":        ["2", "3", "4", "5"],
            }

    def logdir_suffix(ind):
        return "%dl_%du_%.1fd_%.1frd/sub%s" % (grid_params["num_layers"][ind],
                                               grid_params["num_units"][ind],
                                               grid_params["dropout"][ind],
                                               grid_params["var_dropout"][ind],
                                               grid_params["sub"][ind])

    def get_hparams(ind):
        bdrnn = tf.contrib.training.HParams(
            lm_hparams=hparams["cnn_bdlm"],
            num_phyche_features=7,
            num_pssm_features=21,
            num_labels=8,
            embed_units=grid_params["num_units"][ind],
            num_units=grid_params["num_units"][ind],
            num_layers=grid_params["num_layers"][ind],
            filter_size=7,
            residual=False,
            cell_type="lstm",
            out_units=grid_params["num_units"][ind],
            batch_size=50,
            learning_rate=0.2,
            dropout=grid_params["dropout"][ind],
            recurrent_state_dropout=grid_params["var_dropout"][ind],
            recurrent_input_dropout=0.0,
            num_epochs=grid_params["num_epochs"][ind],
            max_gradient_norm=2.,
            max_patience=8,
            eval_step=50,
            model="bdrnn",
            Model=BDRNNModel,
            bdlm_ckpt="/home/dillon/models/thesis/cnn_bdlm/7k_1024f_2l_1024u_256d_noreg_rename/model.ckpt-294001",
            train_file="/home/dillon/data/cpdb/sub"+grid_params["sub"][ind]+"/cpdb_train.tfrecords",
            valid_file="/home/dillon/data/cpdb/sub"+grid_params["sub"][ind]+"/cpdb_valid.tfrecords",
            logdir="/home/dillon/models/thesis/bdrnn/pre/outSEQ/"+logdir_suffix(ind),
            logging=True,
            freeze_bdlm=True,
            input_style="out_SEQ",
            )
        return bdrnn

    for i in range(len(grid_params["num_layers"])):
        exp_hparams = get_hparams(i)
        print(hparams_to_str(exp_hparams))
        print(hparams_to_str(exp_hparams.lm_hparams))
        pretrain(exp_hparams)

if __name__ == "__main__":
    main()
