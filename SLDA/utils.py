import pandas as pd
import numpy as np
import tomotopy as tp
import time
import os
import re
import sys
import logging
from collections import OrderedDict
from itertools import product
from scipy.linalg import solve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


for directory in [PROCESSED_DATA_DIR, SALES_DATA_DIR, TRAINED_MODEL_DIR, EFFICACY_DIR, PLOTS_DIR]:
    if not os.path.exists(directory):
        os.mkdir(directory)
        if directory == PROCESSED_DATA_DIR:
            for document_definition in ['movie', 'review']:
                os.mkdir(os.path.join(directory, document_definition))


def get_labels(corpus):
    model = tp.SLDAModel(corpus=corpus, vars='l')
    return model.docs


def compute_metrics(y, y_pred):
    # convert a new document to its id
    metrics_eval = [mean_absolute_error, mean_squared_error, r2_score]
    metrics_label = ['mae', 'mse', 'r2']
    assert len(y) == len(y_pred), 'Labels and predictions have different lengths!'
    y = to_flatten(y)
    y_pred = to_flatten(y_pred)
    is_valid = (~np.isnan(y)) & (~np.isnan(y_pred))
    assert is_valid.sum() > 0, 'There are no data after dropping nans!'
    data = [f(y_pred[is_valid], y[is_valid]) for f in metrics_eval]
    metrics = pd.DataFrame(data, index=metrics_label)
    return metrics.T


def get_file_suffix(fold_idx=None, **kwargs):
    if fold_idx is None:
        suffix = ''
    else:
        suffix = '_fold_{}'.format(fold_idx)
    for param in ['k', 'tw', 'model_type', 'by', 'label', 'stemmer', 'iterations']:
        if param in kwargs:
            suffix += '_{}_{}'.format(param, kwargs[param])
    return suffix


def extract_words(corpus, doc):
    '''
    # an ad-hoc way to extract processed tokens in tp.Corpus
    :param corpus: tp.Corpus object
    :param doc: tp.Document object
    :return:
    '''
    return [corpus._vocab.id2word[i] for i in doc.words]


def to_flatten(array_like):
    return np.array(array_like).flatten()


def to_2d(array_like):
    return np.array(array_like).reshape((-1, 1))