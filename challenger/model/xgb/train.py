import os
import random

import model.xgb.util as xgboost_util
import xgboost as xgb
from xgboost import plot_importance

import numpy as np
import shap

import matplotlib.pyplot as plt

random.seed(0)

REGRESSION = 0
CLASSIFICATION = 1
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
    n_estimators = 23
    param = {
        'max_depth': 15,
        'booster': 'gbtree',
        "predictor": "gpu_predictor",
        'tree_method': 'gpu_hist',
        'colsample_bytree': 0.85,
        # 'subsample': 0.9,
        'alpha': 2,
        'colsample_bylevel': 0.60,
        'gamma': 0.25,  # 2
    }
    extra_params = dict()

    if problem_type == REGRESSION:
        extra_params.update(
            objective='reg:squarederror',
            eval_metric='mae',
        )
    elif problem_type == CLASSIFICATION:
        scale_pos_weight = 5
        extra_params.update(
            objective='binary:logistic',
            eval_metric=['auc', 'error'],
            base_score=0.5,
            scale_pos_weight=scale_pos_weight
        )
    param.update(extra_params)

    if scaling[TARGET_COLUMN] != 0:
        em = emthresh / scaling[TARGET_COLUMN]

    training = xgb.DMatrix(inputs, outputs, feature_names=data[0][0].columns)
    model = xgb.train(param, training, n_estimators)
    result = {}
    result['train'] = print_performance(
        training_files, model, scaling, problem_type, emthresh=em
    )
    result['test'] = print_performance(
        test_files, model, scaling, problem_type, emthresh=em
    )

    # title = 'classification' if problem_type == CLASSIFICATION else 'regression'
    # plot_importance(model, max_num_features=30)
    # plt.savefig(f'{title}.png', dpi=200, pad_inches=0.5, bbox_inches='tight')

    return result, model
