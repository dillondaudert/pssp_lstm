# do a grid search

import tensorflow as tf
from .van_bdrnn_model import VanillaBDRNNModel
from .hparam_helpers import hparams_to_str
from .pretrain import pretrain

def main():
    grid_params = {
            "num_layers": [2,   2,   3,   3],
            "num_units":  [128, 256, 128, 256],
            "dropout":    [.5,  .5,  .5,  .5],
            "num_epochs": [350, 350, 350, 350],
            }

    def logdir_suffix(ind):
        return "%dl_%du_%.1fd_%.1frd" % (grid_params["num_layers"][ind],
                                         grid_params["num_units"][ind],
                                         grid_params["dropout"][ind],
                                         grid_params["dropout"][ind])

    def get_hparams(ind):
        van_bdrnn = tf.contrib.training.HParams(
            num_phyche_features=7,
            num_pssm_features=21,
            num_labels=8,
            embed_units=grid_params["num_units"][ind],
            num_units=grid_params["num_units"][ind],
            num_layers=grid_params["num_layers"][ind],
            residual=False,
            cell_type="lstm",
            out_units=grid_params["num_units"][ind],
            batch_size=50,
            learning_rate=0.2,
            dropout=grid_params["dropout"][ind],
            recurrent_state_dropout=grid_params["dropout"][ind],
            recurrent_input_dropout=0.0,
            num_epochs=grid_params["num_epochs"][ind],
            max_gradient_norm=2.,
            max_patience=8,
            eval_step=50,
            model="van_bdrnn",
            Model=VanillaBDRNNModel,
            train_file="/home/dillon/data/cpdb/sub1/cpdb_train.tfrecords",
            valid_file="/home/dillon/data/cpdb/sub1/cpdb_valid.tfrecords",
            logdir="/home/dillon/models/thesis/baseline/bdrnn/"+logdir_suffix(ind),
            logging=True,
            )
        return van_bdrnn

    for i in range(len(grid_params["num_layers"])):
        hparams = get_hparams(i)
        print(hparams_to_str(hparams))
        pretrain(hparams)

if __name__ == "__main__":
    main()
