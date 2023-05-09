from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error, accuracy_score,
    f1_score, roc_auc_score,
    log_loss
)
import pandas as pd
import numpy as np
from pandas import concat

REGRESSION = 0
CLASSIFICATION = 1


def tvd(real, pred):
    def get_pdf(data, size, ds_size):
        z = np.zeros(size)
        ind, v = np.unique(data, return_counts=True)
        v = v / ds_size
        np.put(z, ind.astype(int), v)
        return z
    size = int(max(np.max(real), np.max(pred))) + 1
    ds_size = len(real)
    d1 = get_pdf(real, size, ds_size)
    d2 = get_pdf(pred, size, ds_size)
    return 0.5 * np.sum(np.abs(d1 - d2))


def print_metrics(real, prediction, problem_type=REGRESSION, emthresh=3500):
    if problem_type == REGRESSION:
        mse = mean_squared_error(real, prediction)
        mae = mean_absolute_error(real, prediction)
        r2 = r2_score(real, prediction)

        scores = {'mse': mse, 'mae': mae, 'r2': r2}

    elif problem_type == CLASSIFICATION:
        pred = np.round(prediction)
        scores = {
            'accuracy': accuracy_score(real, pred),
            'f1': f1_score(real, pred),
        }

    return scores


def calculate_scaling(training_paths):
    scaling = {}
    # calculate scaling factors
    for f in training_paths:
        df = pd.read_csv(f, index_col=False)
        for column in df.columns:
            if column not in scaling:
                scaling[column] = 0.
            scaling[column] = max(scaling[column], float(df[column].max()))
            if scaling[column] == 0.0:
                scaling[column] = 1.0
    return scaling


def resize(s, scaling):
    return s/scaling[s.name]


def prepare_files(files, window_size, scaling, target_column='size'):
    result = []

    for f in files:
        df = pd.read_csv(f, index_col=False)
        if scaling is not None:
            df = df.apply((lambda x: resize(x, scaling)), axis=0)
        flow_size = df[target_column]
        df[target_column] = flow_size
        # extend the window
        final_df = df.copy()
        for sample_num in range(1, window_size):
            shifted = df.shift(sample_num)
            shifted.columns = map(lambda x: x+str(sample_num), shifted.columns)
            final_df = concat([shifted, final_df], axis=1)

        final_df = final_df.fillna(0)
        final_df = final_df.drop(target_column, axis=1)

        result.append((final_df, flow_size))
    return result


def make_io(data):
    inputs = None
    outputs = None
    for d in data:
        i_data = d[0].values
        o_data = d[1].tolist()
        if inputs is None:
            inputs = i_data
            outputs = o_data
        else:
            inputs = np.append(inputs, i_data, axis=0)
            outputs = np.append(outputs, o_data)
    return (inputs, outputs)
