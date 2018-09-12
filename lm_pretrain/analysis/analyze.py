import pandas as pd
import numpy as np
from .metrics import accuracy, accuracy_vs_len
from .data_loader import load_data
from ..lookup import STRUCT_ALPHABET

def analyze(datafiles: list, outfile: str):
    res_dfs = []
    for f in datafiles:
        print("Loading data from %s" % f)
        df = load_data(f)

        y_pred = df.logits.apply(lambda x: np.argmax(x, axis=-1))
        y_true = df.ss.apply(lambda x: np.argmax(x, axis=-1))

        acc_df = accuracy(y_true, y_pred, classes=STRUCT_ALPHABET)
        acc_len_df = accuracy_vs_len(y_true, y_pred, df.len, boundaries=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])

        res_df = pd.concat([acc_df, acc_len_df])
        res_df.columns = [df.dataset.iloc[0]+"_accuracy", df.dataset.iloc[0]+"_count"]
        res_dfs.append(res_df)

    res_df = pd.concat(res_dfs, axis=1)
    print(res_df)
    res_df.to_csv(outfile)
