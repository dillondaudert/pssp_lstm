# load prediction data into dataframe

import pandas as pd
import numpy as np
import re

seq_cols = ["seq", "phyche", "pssm", "logits", "ss"]
seq_hcols = ["h_0", "h_1", "h_2", "lm_logits"]
title_cols = ["dataset", "id", "len", "position"]
out_seq_cols = ["amino", "phyche", "pssm", "logits", "ss"]

def load_data(path: str) -> pd.DataFrame:
    raw_data = pd.read_pickle(path)

    def get_dataset(datafile):
        for name in ["train", "valid", "test"]:
            if re.search(name, datafile) is not None:
                for sub in ["sub1", "sub2", "sub3", "sub4", "sub5"]:
                    if re.search(sub, datafile) is not None:
                        return sub+"_"+name


        raise Exception()

    recs = []
    if np.isscalar(raw_data["h_0"].iloc[0]):
        itercols = seq_cols
        out_cols = out_seq_cols
    else:
        itercols = seq_cols+seq_hcols
        out_cols = out_seq_cols+seq_hcols

    for i in range(raw_data.shape[0]):

        sample_recs = [tuple([get_dataset(path), raw_data.iloc[i].id, raw_data.iloc[i].len, j] + \
                             [raw_data.iloc[i][col][j, :] for col in itercols]) for j in range(raw_data.iloc[i].len)]
        recs = recs + sample_recs

    out_df = pd.DataFrame.from_records(data=recs, columns=title_cols+out_cols)

    return out_df
