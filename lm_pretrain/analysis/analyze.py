import pandas as pd
import numpy as np
from .metrics import accuracy
from .data_loader import load_data
from ..lookup import STRUCT_ALPHABET

def analyze(datafile: str, outfile: str):
    print("Loading data from {datafile}")
    df = load_data(datafile)

    y_pred = df.logits.apply(lambda x: np.argmax(x, axis=-1))
    y_true = df.ss.apply(lambda x: np.argmax(x, axis=-1))

    print(accuracy(y_pred, y_true, classes=STRUCT_ALPHABET))
