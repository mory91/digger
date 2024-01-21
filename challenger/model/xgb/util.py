from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    mean_absolute_percentage_error
)
import pandas as pd
import numpy as np
from pandas import concat

REGRESSION = 0
CLASSIFICATION = 1
REGCLASS = 2


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


def smape(A, F):
    a = np.array(A)
    f = np.array(F)
    return 100/len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)))


def print_metrics(real, prediction, problem_type=REGRESSION, emthresh=3500):
    if problem_type == REGRESSION:
        mse = mean_squared_error(real, prediction)
        mae = mean_absolute_error(real, prediction)
        r2 = r2_score(real, prediction)
        mape = mean_absolute_percentage_error(real, prediction)
        smapes = smape(real, prediction)

        scores = {'mse': mse, 'mae': mae, 'r2': r2, 'mape': mape, 'smape': smapes}

    elif problem_type in (CLASSIFICATION, REGCLASS):
        pred = np.round(prediction)
        scores = {
            'accuracy': accuracy_score(real, pred),
            'f1': f1_score(real, pred),
        }

    return scores


def calculate_scaling(training_path):
    scaling = {}
    # calculate scaling factors
    df = pd.read_csv(training_path, index_col=False)
    for column in df.columns:
        if column not in scaling:
            scaling[column] = 0.
        scaling[column] = max(scaling[column], float(df[column].max()))
        if scaling[column] == 0.0:
            scaling[column] = 1.0
    mean, std = df.mean(), df.std()
    return scaling, mean, std


def resize(s, scaling, range_min=0, range_max=1):
    if scaling[s.name] is None:
        range_max = 1
        range_min = 0
        nums_max = 1
        nums_min = 0
    else:
        nums_max = scaling[s.name]
        nums_min = 0
    return (((s - nums_min)/(nums_max - nums_min)) * (range_max - range_min)) + range_min


def prepare_files(files, window_size, scaling, mean, std, target_column='size'):
    result = []

    df = pd.read_csv(files, index_col=False)
    # Convert to categorical here
    # if 'src_port' in df:
    #     df['src_port'] = df['src_port'].astype('category')
    if scaling is not None:
        df = df.apply((lambda x: resize(x, scaling, 1, 3)), axis=0)
    flow_size = df[target_column]
    df[target_column] = flow_size
    # df[df.columns] = (df - mean) / std
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
