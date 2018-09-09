
import pandas as pd
from sklearn.metrics import log_loss
from typing import List, Optional

__all__ = ["cross_entropy_loss", "cross_entropy_loss_vs_len", "cross_entropy_loss_vs_pos", "accuracy", "accuracy_vs_len", "accuracy_vs_pos", "confusion_matrix", "_class_accuracy", "_bin_accuracy"]

def cross_entropy_loss(y_true: pd.Series,
                       logits: pd.Series,
                       classes: Optional[List[str]] = None) -> pd.DataFrame:
    pass

def cross_entropy_loss_vs_len(y_true: pd.Series,
                              logits: pd.Series,
                              lens: pd.Series,
                              classes: Optional[List[str]] = None) -> pd.DataFrame:
    pass

def cross_entropy_loss_vs_pos(y_true: pd.Series,
                              logits: pd.Series,
                              lens: pd.Series,
                              res_pos: pd.Series,
                              classes: Optional[List[str]] = None) -> pd.DataFrame:
    pass

def _class_accuracy(y_true: pd.Series,
                    y_pred: pd.Series,
                    cls: int) -> float:
    y_true_cls = y_true[y_true == cls]
    y_pred_cls = y_pred[y_true == cls]
    return (y_true_cls == y_pred_cls).mean()

def accuracy(y_true: pd.Series,
             y_pred: pd.Series,
             classes: Optional[List[str]] = None) -> pd.DataFrame:
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) == 1

    if classes is not None:
        num_classes = len(classes)
        index = classes+["Total"]
        assert y_true.max()+1 <= num_classes
        assert y_pred.max()+1 <= num_classes
    else:
        num_classes = max(y_true.max(), y_pred.max())+1
        index = None

    acc = [_class_accuracy(y_true, y_pred, i) for i in range(num_classes)]
    acc = acc + [(y_true == y_pred).mean()]

    return pd.DataFrame(data={"accuracy": acc},
                        index=index)

def _bin_accuracy(y_true: pd.Series,
                  y_pred: pd.Series,
                  lens: pd.Series,
                  bin_start: int,
                  bin_end: int) -> float:
    y_true_rng = y_true[(lens >= bin_start) & (lens < bin_end)]
    y_pred_rng = y_pred[(lens >= bin_start) & (lens < bin_end)]
    return (y_true_rng == y_pred_rng).mean()

def accuracy_vs_len(y_true: pd.Series,
                    y_pred: pd.Series,
                    lens: pd.Series,
                    bins: Optional[int] = None,
                    classes: Optional[List[str]] = None) -> pd.DataFrame:
    # create bin boundaries
    if bins is None:
        width = 1
    else:
        width = lens.max()//bins

    boundaries = [i for i in range(lens.min(), lens.max()+width+1, width)]

    # calc accuracy for each bin
    acc = [_bin_accuracy(y_true, y_pred, lens, boundaries[i], boundaries[i+1]) \
            for i in range(len(boundaries)-1)]

    index = [boundaries[i] for i in range(1, len(boundaries))]

    return pd.DataFrame(data={"accuracy": acc},
                        index=index)

def accuracy_vs_pos(y_true: pd.Series,
                    y_pred: pd.Series,
                    lens: pd.Series,
                    res_pos: pd.Series,
                    classes: Optional[List[str]] = None) -> pd.DataFrame:
    pass

def confusion_matrix(y_true: pd.Series,
                     y_pred: pd.Series,
                     classes: Optional[List[str]] = None,
                     normalized: bool = True) -> pd.DataFrame:
    pass
