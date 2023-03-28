import os
import random

import xgboost as xgb
import xgboost_util

import numpy as np

random.seed(0)

REGRESSION = 0
CLASSIFICATION = 1
NUMBER_OF_TREES = 50
WINDOW_SIZE = 5
TARGET_COLUMN = 'size'


def print_performance(
    files, model, scaling, problem_type=REGRESSION, emthresh=3500
):
    real = []
    predicted = []
    for f in files:
        data = xgboost_util.prepare_files(
            [f],
            WINDOW_SIZE,
            scaling,
            TARGET_COLUMN
        )
        inputs, outputs = xgboost_util.make_io(data)
        y_pred = model.predict(
            xgb.DMatrix(inputs, feature_names=data[0][0].columns)
        )
        pred = y_pred.tolist()

        real += outputs
        predicted += pred

    return xgboost_util.print_metrics(
        real, predicted, problem_type, emthresh=emthresh
    )


def xgboost_learn(
    training_path, test_path, problem_type=REGRESSION, emthresh=3500
):
    training_files = [
        os.path.join(training_path, f) for f in os.listdir(training_path)
    ]
    test_files = [
        os.path.join(test_path, f) for f in os.listdir(test_path)
    ]
    scaling = xgboost_util.calculate_scaling(training_files)
    data = xgboost_util.prepare_files(
        training_files,
        WINDOW_SIZE,
        scaling,
        TARGET_COLUMN
    )

    inputs, outputs = xgboost_util.make_io(data)

    # fit model no training data
    param = {
        'max_depth': 10,
        'booster': 'gbtree',
        "predictor": "gpu_predictor",
        'tree_method': 'gpu_hist'
    }
    extra_params = dict()

    if problem_type == REGRESSION:
        extra_params.update(
            objective='reg:squarederror',
            eval_metric='mae',
            base_score=2
        )
    elif problem_type == CLASSIFICATION:
        scale_pos_weight = (
            (np.array(outputs) == 0).sum() / (np.array(outputs) == 1).sum()
        )
        extra_params.update(
            objective='binary:logistic',
            eval_metric=['auc', 'error'],
            base_score=0.5,
            scale_pos_weight=scale_pos_weight
        )

    param.update(extra_params)

    em = emthresh / scaling[TARGET_COLUMN]

    training = xgb.DMatrix(inputs, outputs, feature_names=data[0][0].columns)
    model = xgb.train(param, training, NUMBER_OF_TREES)
    result = {}
    result['train'] = print_performance(
        training_files, model, scaling, problem_type, emthresh=em
    )
    result['test'] = print_performance(
        test_files, model, scaling, problem_type, emthresh=em
    )

    return result, model
