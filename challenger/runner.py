import json
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from constants import NANO_TO_MICRO, ALL_FEATURES
from dataset import FullDataset
from model.flux import FluxClassifier, FluxRegression
from feature.subset import (
    get_all_features,
    get_nsdi_features,
)

BACKUP = "/usr/local/etc/backup/"
PREFIX = "../data/9/node-1"
NODE_1 = 237
NODE_2 = 229
NODE_3 = 212
NODE_4 = 144
MORTEZA = 222
BARDIA = 116

A_NODE_1 = 161
A_NODE_2 = 128
A_NODE_3 = 45

ME = NODE_1
OTHERS = [NODE_4, NODE_2, NODE_3]

logging.basicConfig(filename='out.log', encoding='utf-8', level=logging.INFO)


class Runner:
    def __init__(self, times, prefix,
                 source_ip, destination_ip, full_features):
        self.times = times
        self.prefix = prefix
        self.source_ip = source_ip
        self.destination_ip = destination_ip
        self.full_features = full_features

    def run(self):
        result = {}
        for td in self.times:
            logging.info(td)

            if td not in result:
                result[td] = {}
            time_delta = td * NANO_TO_MICRO
            ds = FullDataset(
                time_delta, self.prefix,
                self.source_ip, self.destination_ip,
                self.full_features
            )
            ds = ds.create_ds()

            # limit
            # ds = ds.iloc[:10_000_000]
            # max_size = max(7_000_000, len(ds) * 0.8)
            ds = ds.iloc[:600_000]

            train_ds = get_nsdi_features(ds)
            nsdi_models = [
                ('nsdi_regression', FluxRegression),
                ('nsdi_classifier', FluxClassifier),
            ]
            for k, model_cls in nsdi_models:
                m = model_cls(train_ds)
                result[td][k] = m.train()

            train_ds = get_all_features(ds)
            all_models = [
                ('all_regression', FluxRegression),
                ('all_classifier', FluxClassifier)
            ]
            for k, model_cls in all_models:
                m = model_cls(train_ds)
                result[td][k] = m.train()

            logging.info(f"{td}, {len(ds)}")

        return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'r':
        results = json.load(open('result.json'))
        results_dict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
        # metric test/train nsdi/all
        times = sorted(list(map(int, results.keys())))
        metrics = set()
        for time in times:
            for setup, setup_result in results[str(time)].items():
                for valid, valid_result in setup_result.items():
                    for metric, metric_result in valid_result.items():
                        whos, ptype = setup.split('_')
                        results_dict[ptype][metric][whos][valid].append(
                            metric_result
                        )

        results_dict = dict(results_dict)
        for ptype in results_dict:
            for metric in results_dict[ptype]:
                nsdi_test = results_dict[ptype][metric]['nsdi']['test']
                nsdi_train = results_dict[ptype][metric]['nsdi']['train']
                full_test = results_dict[ptype][metric]['all']['test']
                full_train = results_dict[ptype][metric]['all']['train']
                plt.xlabel("Flowlet Time Threshold (Microseconds)")
                plt.ylabel(metric)
                plt.title(f"Flow {ptype} Score (SGD W - M)")
                plt.plot(
                    times, full_train, 'b--', label='Our features train score'
                )
                plt.plot(
                    times, full_test, 'b-', label='Our features test score'
                )
                plt.plot(
                    times, nsdi_train, 'r--', label='NSDI features train score'
                )
                plt.plot(
                    times, nsdi_test, 'r-', label='NSDI features train score'
                )
                plt.legend(loc='best')
    elif len(sys.argv) > 1 and sys.argv[1] == 'cdf':
        paths = [
            'sgd-node-3-node-4-6-hours-backup/',
            'tf-alexnet-node-1-3-hours-backup/',
            'tf-alexnet-node-3-node-4-3-hours-backup/',
            'tf-alexnet-node-3-node-4-9-hours-backup/'
        ]
        times = range(500, 10200, 250)
        for p in paths:
            for t in times:
                flows = pd.read_csv(f"{p}/{t}/flows.csv")['size'].values
                hist, bin_edges = np.histogram(flows)
                cdf = np.cumsum(hist)
                print(cdf)
    else:
        times = range(31000, 40200, 1000)
        source, destinations = ME, OTHERS
        prefix = PREFIX
        runner = Runner(times, prefix, source, destinations, ALL_FEATURES)
        result = runner.run()
        result_file = open('result.json', 'w')
        json.dump(result, result_file)
