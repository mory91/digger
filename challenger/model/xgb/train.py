import os
import random

import model.xgb.util as xgboost_util
import xgboost as xgb

import catboost as ctb
import numpy as np


random.seed(0)

REGRESSION = 0
CLASSIFICATION = 1
REGCLASS = 2
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
    scaling, mean, std = xgboost_util.calculate_scaling(training_files)
    if scaling[TARGET_COLUMN] != 0:
        em = emthresh / scaling[TARGET_COLUMN]
    if problem_type == CLASSIFICATION:
        scaling[TARGET_COLUMN] = None
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
        'alpha': 2,
        'colsample_bylevel': 0.60,
        'gamma': 0.25,
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

    training = xgb.DMatrix(inputs, outputs, enable_categorical=True, feature_names=data[0][0].columns)
    model = xgb.train(param, training, n_estimators)
    result = {}
    result['train'] = print_performance(
        training_files, model, scaling, problem_type, emthresh=em
    )
    result['test'] = print_performance(
        test_files, model, scaling, problem_type, emthresh=em
    )

    return result, model


def catboost_learn(
    training_path, test_path, problem_type=REGRESSION, emthresh=15000
):
    training_files = os.path.join(training_path, os.listdir(training_path)[0])
    test_files = os.path.join(test_path, os.listdir(training_path)[0])
    scaling, mean, std = xgboost_util.calculate_scaling(training_files)
    if scaling[TARGET_COLUMN] != 0:
        em = ((emthresh / scaling[TARGET_COLUMN]) * 2) + 1
    if problem_type == CLASSIFICATION:
        scaling[TARGET_COLUMN] = None
    data = xgboost_util.prepare_files(
        training_files,
        WINDOW_SIZE,
        scaling,
        mean,
        std,
        TARGET_COLUMN
    )
    inputs, outputs = xgboost_util.make_io(data)

    if problem_type == REGRESSION:
        model = ctb.CatBoostRegressor(
            logging_level="Silent",
            task_type="GPU",
            devices="0:1",
        )
    elif problem_type == CLASSIFICATION:
        model = ctb.CatBoostClassifier(
            logging_level="Silent",
            task_type="GPU",
            devices="0:1",
        )
        outputs = [int(o) for o in outputs]

    elif problem_type == REGCLASS:
        model = ctb.CatBoostRegressor(
            logging_level="Silent",
            task_type="GPU",
            devices="0:1",
        )

    # categorical values
    model.set_feature_names(list(data[0][0].columns))
    # import time
    # start = time.process_time()
    model.fit(inputs, outputs)
    # print(problem_type, time.process_time() - start)

    result = {}
    result['train'] = ctb_print_performance(
        training_files, model, scaling, mean, std, problem_type, emthresh=em
    )
    result['test'] = ctb_print_performance(
        test_files, model, scaling, mean, std, problem_type, emthresh=em
    )

    return result, model


def ctb_print_performance(
    file, model, scaling, mean, std, problem_type=REGRESSION, emthresh=3500
):
    real = []
    predicted = []
    data = xgboost_util.prepare_files(
        file,
        WINDOW_SIZE,
        scaling,
        mean,
        std,
        TARGET_COLUMN
    )
    inputs, outputs = xgboost_util.make_io(data)
    prediction_args = {
        'data': inputs
    }
    if problem_type == CLASSIFICATION:
        prediction_args.update(prediction_type='Class')
    y_pred = model.predict(**prediction_args)
    if problem_type == REGCLASS:
        y_pred = (y_pred > emthresh) * 1
        outputs = list((np.array(outputs) > emthresh) * 1)
    pred = y_pred.tolist()

    real += outputs
    predicted += pred

    return xgboost_util.print_metrics(
        real, predicted, problem_type, emthresh=emthresh
    )
