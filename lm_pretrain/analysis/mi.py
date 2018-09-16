import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from typing import Optional, List

def mutual_info(target: pd.Series,
                variables: pd.Series,
                n_neighbors: Optional[int] = 5,
                labels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Return a DataFrame of the estimated mutual information between
    the target and the variables.
    `variables` should be a series (n_samples, 1) of numpy arrays,
    each of shape (n_features,)
    `target` should be a series (n_samples,) of the target variable
    value for each sample.
    `n_neighbors` is the number of neighbors to use for the MI 
    estimator
    If a list of labels is provided, it will be used as the index
    of the resulting DataFrame.
    """
    # concatenate into a matrix (n_samples, n_features)
    if labels is not None:
        assert len(labels) == variables.iloc[0].shape[0]
    xs = np.concatenate([variables.iloc[i].reshape(1, -1) for i in range(variables.shape[0])])
    ys = target.values
    
    mi = mutual_info_classif(X=xs, y=ys, discrete_features=False, n_neighbors=n_neighbors)
    
    df = pd.DataFrame.from_dict(data={"mutual info": mi})
    df.reindex(labels=labels)