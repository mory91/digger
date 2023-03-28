import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from trace_process import *
import xgboost_learn

logging.basicConfig(level=logging.INFO)

IPS = [237, 212, 144]
MASTER = 237
NODE_2 = 229
NODE_3 = 212

NUMBER_OF_TREES = 50
SHIFT_WINDOW = 5

REGRESSION = 0
CLASSIFICATION = 1

SPARK_PREFIX1 = "../data/25/node-2"
SPARK_PREFIX2 = "../data/25/node-3"
SPARK_PREFIX3 = "../data/26/node-2"
SPARK_PREFIX4 = "../data/26/node-3"
SPARK_PREFIX5 = "../data/27/node-2"
SPARK_PREFIX6 = "../data/27/node-3"
SPARK_PREFIX7 = "../data/28/node-2"
SPARK_PREFIX8 = "../data/28/node-3"
SPARK_PREFIX9 = "../data/29/node-2"
SPARK_PREFIX10 = "../data/29/node-3"

ip_pairs = [(NODE_2, MASTER), (NODE_3, MASTER), (NODE_2, MASTER), (NODE_3, MASTER), (NODE_2, MASTER), (NODE_3, MASTER), (NODE_2, MASTER), (NODE_3, MASTER)]

TENSORFLOW_PREFIX1 = "../data/24/node-1"
TENSORFLOW_PREFIX2 = "../data/30/node-1"

SPARK_PREFIX10 = "../data/29/node-3"

MEMCACHED_PREFIX1 = "../data/31/node-1"

PACKETS = "packets"
ALLOCS = "allocations"
DISK_READS = "disk_read"
DISK_WRITES = "disk_write"
VIRTUAL_MEMORY = "memory"
RSS_MEMORY = "rss_memory"
DATA_MEMORY = "data_memory"
S_TIME = "s_time"
U_TIME = "u_time"

TRAIN = "train"
TEST = "test"

CUM_TRACES = [ALLOCS]

def to_em(outputs):
    em_outputs = []
    for idx in range(len(outputs)):
        em_outputs.append(np.array(outputs[idx] > np.median(outputs[idx])) * 1.0)
    return em_outputs

def create_long_dataset(packets_df, trace_names, time_delta):
    flows_packets_df = extract_flow_by_ip(packets_df, TARGET_IP, MASTER)[0]

    start_time = packets_df['timestamp'].values[0]
    end_time = packets_df['timestamp'].values[-1]
    time_slots = np.arange(start_time, end_time, time_delta)
    time_slots = np.column_stack([time_slots, np.roll(time_slots, -1)])[:-2]

    src_ip, dest_ip = TARGET_IP, MASTER
    in_data = packets_df[(packets_df['src_ip'] == src_ip) & (packets_df['dest_ip'] == dest_ip)][['timestamp', 'size']].values
    out_data = packets_df[(packets_df['src_ip'] == dest_ip) & (packets_df['dest_ip'] == src_ip)][['timestamp', 'size']].values
    in_data[:, 1] = np.cumsum(in_data[:, 1])
    out_data[:, 1] = np.cumsum(out_data[:, 1])
    vals_in_in = np.searchsorted(in_data[:, 0], time_slots, side='left')
    vals_in_in[vals_in_in == len(in_data)] = len(in_data) - 1
    ins = in_data[vals_in_in, 1]
    ins = ins[:, 1] - ins[:, 0]
    vals_in_out = np.searchsorted(out_data[:, 0], time_slots, side='left')
    vals_in_out[vals_in_out == len(out_data)] = len(out_data) - 1
    outs = out_data[vals_in_out, 1]
    outs = outs[:, 1] - outs[:, 0]

    ds = pd.DataFrame({'size': outs, 'network_in': ins, 'time': time_slots[:, 0]})
    
    CUM_TRACES = [ALLOCS]

    agg_record_results = {}
    for trace_name in trace_names:
        sep = ','
        if ALLOCS in trace_name:
            sep = '\t'
        fts = time_slots[:, 0]
        _df = get_trace(trace_name, sep=sep, names=['timestamp', 'size'], dtype={})
        values_in_df = np.searchsorted(_df['timestamp'].values, fts)
        values_in_df[values_in_df == len(_df)] = len(_df) - 1

        if (any([c in trace_name for c in CUM_TRACES])):
            size_csum = np.cumsum(_df['size'])
            aggregated_record_in_flow = size_csum[values_in_df].values
        else:
            aggregated_record_in_flow = _df['size'][values_in_df].values
        
        agg_record_results[trace_name] = aggregated_record_in_flow
    
    for trace_name in agg_record_results:
        t_name = trace_name.split('/')[-1]
        trace_ds = pd.DataFrame({t_name: agg_record_results[trace_name]})
        ds = pd.concat([ds, trace_ds], axis=1)

    return ds

def create_flow_centric_dataset(packets_df, features_path, target_column, time_delta, source, destination, output_type=REGRESSION, emthresh=3500, init_cum=None, in_init=0, out_init=0):
    flow_indicator = ['src_ip', 'dest_ip']
    flows_packets_df = extract_flow_by_ip(packets_df, source, destination)

    flows_times, flows_sizes = [], []
    flows_gaps = []
    flows_networkins = []
    flows_networkouts = []
    for idx in range(len(flows_packets_df)):
        flow_time, flow_size = get_flow(flows_packets_df[idx].values, time_delta)
        flow_gap = np.roll(np.roll(flow_time[:, 0], -1) - flow_time[:, 1], 1)

        fts = flow_time[:, 0]
        in_agg, out_agg = get_net_in_out(packets_df, fts, source, destination, in_init_value=in_init, out_init_value=out_init)

        flows_times.append(flow_time)
        flows_sizes.append(flow_size)
        flows_gaps.append(flow_gap)
        flows_networkins.append(in_agg)
        flows_networkouts.append(out_agg)
    
    agg_record_results = {}
    for trace_name in features_path:
        sep = ','
        if ALLOCS in trace_name:
            sep = '\t'

        _df = get_trace(trace_name, sep=sep, names=['timestamp', 'size'], dtype={})
        agg_record_results[trace_name] = []
        for idx in range(len(flows_times)):
            fts = flows_times[idx][:, 0]
        
            values_in_df = np.searchsorted(_df['timestamp'].values, fts)
            values_in_df[values_in_df == len(_df)] = len(_df) - 1

            if (any([c in trace_name for c in CUM_TRACES])):
                size_csum = np.cumsum(_df['size'])
                aggregated_record_in_flow = size_csum[values_in_df].values
                if trace_name in init_cum:
                    aggregated_record_in_flow = aggregated_record_in_flow + init_cum[trace_name]
            else:
                aggregated_record_in_flow = _df['size'][values_in_df].values
            
            agg_record_results[trace_name].append(aggregated_record_in_flow)
    
    datasets = []
    for idx in range(len(flows_gaps)):
        o = flows_sizes[idx]
        # EDIT
        if output_type == CLASSIFICATION:
            o = np.array(o > emthresh) * 1.0
        ds = pd.DataFrame({'start_time': flows_times[idx][:, 0], 'size': o, 'gap': flows_gaps[idx], 'networkin': flows_networkins[idx], 'networkout': flows_networkouts[idx]})[1:]
        for trace_name in agg_record_results:
            t_name = trace_name.split('/')[-1]
            trace_ds = pd.DataFrame({t_name: agg_record_results[trace_name][idx]})
            ds = pd.concat([ds, trace_ds], axis=1)
        datasets.append(ds)
    
    
    outputs = []
    feature_names = []
    """
    for idx in range(len(datasets)):
        datasets[idx], output, columns = transform_dataset(datasets[idx], target_column)
        outputs.append(output)
        feature_names.append(columns)
    """

    for idx in range(len(datasets)):
        datasets[idx] = datasets[idx].fillna(0)

    return datasets, outputs, feature_names


def get_net_in_out(packets_df, flow_times, src_ip, dest_ip, in_init_value=0, out_init_value=0):
    in_data = packets_df[(packets_df['src_ip'] == src_ip) & (packets_df['dest_ip'] == dest_ip)][['timestamp', 'size']].values
    out_data = packets_df[(packets_df['src_ip'] == dest_ip) & (packets_df['dest_ip'] == src_ip)][['timestamp', 'size']].values
    in_data[:, 1] = np.cumsum(in_data[:, 1])
    out_data[:, 1] = np.cumsum(out_data[:, 1])
    vals_in_in = np.searchsorted(in_data[:, 0], flow_times)
    vals_in_in[vals_in_in == len(in_data)] = len(in_data) - 1
    in_agg = in_data[vals_in_in, 1]
    vals_in_out = np.searchsorted(out_data[:, 0], flow_times)
    vals_in_out[vals_in_out == len(out_data)] = len(out_data) - 1
    out_agg = out_data[vals_in_out, 1]
    return in_agg + in_init_value, out_agg + out_init_value

def transform_dataset(ds, target_column):
    main_ds = ds.copy()
    for s in range(1, SHIFT_WINDOW):
        shifted_ds = main_ds.shift(s).fillna(0)
        shifted_ds.columns = map(lambda x: x + str(s), shifted_ds.columns)
        ds = pd.concat([shifted_ds, ds], axis=1)
    output = ds[target_column].fillna(0).values
    ds = ds.fillna(0).drop(target_column, axis=1)
    columns = ds.columns
    ds = MinMaxScaler().fit_transform(ds)
    return ds, output, columns

def get_scores(y_t, y_pred):
    mse = mean_squared_error(y_t, y_pred)
    mae = mean_absolute_error(y_t, y_pred)
    r2 = r2_score(y_t, y_pred)
    return mse, mae, r2

def get_c_scores(y_t, y_pred):
    return accuracy_score(y_t, y_pred)

def train_regression_xgb_model(datasets, outputs, feature_names):
    param = {
        'max_depth' : 10,
        'objective' : 'reg:squarederror',
        'booster' : 'gbtree',
        'base_score' : 2,
        'eval_metric': 'mae',
        'predictor': 'gpu_predictor',
        'tree_method': 'gpu_hist'
    }
    models = []
    scores = []
    for idx in range(len(datasets)):
        train_mat = xgb.DMatrix(datasets[idx], outputs[idx], feature_names=feature_names[idx])
        model = xgb.train(param, train_mat, NUMBER_OF_TREES)
        pred = model.predict(train_mat)
        mse, mae, r2 = get_scores(outputs[idx], pred)
        scores.append([mse, mae, r2])
        models.append(model)
        logging.info(f"R2: {r2}")
    
    return models, scores

def train_classification_xgb_model(datasets, outputs, feature_names):
    max_rounds = 5
    params = {
        'objective':'binary:logistic',
        'eval_metric':['auc', 'error'],
    }
    models = []
    scores = []
    for idx in range(len(datasets)):
        xgb_cl = xgb.XGBClassifier(n_estimators=max_rounds, **params)
        xgb_cl.fit(datasets[idx], outputs[idx], verbose=True)
        pred = xgb_cl.predict(datasets[idx])
        acc = get_c_scores(outputs[idx], pred)
        scores.append([acc])
        logging.info(f"ACC: {acc}")
        models.append(xgb_cl)
    
    return models, scores

def evaluate_classification_model(models, datasets, outputs, feature_names):
    scores = []
    for idx in range(len(datasets)):
        pred = models[idx].predict(datasets[idx])
        acc = get_c_scores(outputs[idx], pred)
        scores.append([acc])
    return scores

def evaluate_regression_model(models, datasets, outputs, feature_names):
    scores = []
    for idx in range(len(datasets)):
        test_mat = xgb.DMatrix(datasets[idx], outputs[idx], feature_names=feature_names[idx])
        pred = models[idx].predict(test_mat)
        mse, mae, r2 = get_scores(outputs[idx], pred)
        scores.append([mse, mae, r2])
    return scores

def run_flux_regression(prefixes, files, output_type=REGRESSION):
    deltas = list(range(500, 20000, 1000))
    all_train_scores, all_test_scores = [], []
    for td in deltas:
        time_delta = td * NANO_TO_MICRO
        target_column = 'size'

        for idx, prefix in enumerate(prefixes):
            # Create paths
            paths = []
            source, destination = ip_pairs[idx]
            for f in files:
                paths.append(f"{prefix}/{TRAIN}/{f}")
            packets_path = f"{prefix}/{TRAIN}/{PACKETS}"

            packets_df = get_trace(packets_path, names=['timestamp', 'size', 'src_ip', 'dest_ip', 'src_port', 'dest_port', 'dir'], dtype={})
            packets_df = packets_df[packets_df['src_ip'].isin([source, destination]) & packets_df['dest_ip'].isin([source, destination])]

            logging.info(f"Creating train dataset for {time_delta}")

            packets_train, packets_test = train_test_split(packets_df, shuffle=False, test_size=0.2)
            train_datasets, train_outputs, feature_names = create_flow_centric_dataset(packets_train, paths, target_column, time_delta, source, destination, output_type=output_type)
            test_datasets, test_outputs, feature_names = create_flow_centric_dataset(packets_test, paths, target_column, time_delta, source, destination, output_type=output_type)

            name = ''.join(prefix.split('/')[-2:]).replace('/', '_')
            train_datasets[0].to_csv(f"now/train/{name}.csv", index=False)
            test_datasets[0].to_csv(f"now/test/{name}.csv", index=False)

        # NORMALIZE TIME
        # FIND MISSED DATA
        r, m = xgboost_learn.xgboost_learn(f'now/train/', f'now/test/', problem_type=output_type)
        all_test_scores.append(r['test'])
        all_train_scores.append(r['train'])
        
        """
        models, train_scores = train_regression_xgb_model(train_datasets, train_outputs, feature_names)
        all_train_scores.append(train_scores)
        logging.info(f"TRAIN SCORES: {train_scores}")

        scores = evaluate_regression_model(models, test_datasets, test_outputs, feature_names)
        all_test_scores.append(scores)
        logging.info(f"TEST SCORES: {scores}")
        """

    return deltas, all_test_scores, all_train_scores

def run_flux_classification(prefix, files):
    deltas = list(range(500, 20000, 1000))
    all_train_scores, all_test_scores = [], []
    for td in deltas:
        time_delta = td * 1e9
        target_column = 'size'

        # Create paths
        train_paths = []
        test_paths = []
        for f in files:
            train_paths.append(f"{prefix}/{TRAIN}/{f}")
            test_paths.append(f"{prefix}/{TEST}/{f}")
        packets_train = f"{prefix}/{TRAIN}/{PACKETS}"
        packets_test = f"{prefix}/{TEST}/{PACKETS}"

        packets_train_df = get_trace(packets_train, names=['timestamp', 'size', 'src_ip', 'dest_ip', 'src_port', 'dest_port', 'dir'], dtype={'dir': 'int8'})
        packets_train_df = packets_train_df[packets_train_df['src_ip'].isin([MASTER, TARGET_IP]) & packets_train_df['dest_ip'].isin([MASTER, TARGET_IP])]
        logging.info(f"Creating train dataset for {time_delta}")
        train_datasets, train_outputs, feature_names = create_flow_centric_dataset(packets_train_df, train_paths, target_column, time_delta, CLASSIFICATION)

        packets_test_df = get_trace(packets_test, names=['timestamp', 'size', 'src_ip', 'dest_ip', 'src_port', 'dest_port', 'dir'], dtype={'dir': 'int8'})
        packets_test_df = packets_test_df[packets_test_df['src_ip'].isin([MASTER, TARGET_IP]) & packets_test_df['dest_ip'].isin([MASTER, TARGET_IP])]
        test_datasets, test_outputs, feature_names = create_flow_centric_dataset(packets_test_df, test_paths, target_column, time_delta, CLASSIFICATION)

        """
        test_outputs = to_em(test_outputs)
        train_outputs = to_em(train_outputs)
        """

        train_datasets[0].to_csv('now/train/f.csv', index=False)
        test_datasets[0].to_csv('now/test/f.csv', index=False)
        r, m = xgboost_learn.xgboost_learn(f'now/train/', f'now/test/', CLASSIFICATION)
        all_test_scores.append(r['test'])
        all_train_scores.append(r['train'])

        """
        models, train_scores = train_classification_xgb_model(train_datasets, train_outputs, feature_names)
        all_train_scores.append(train_scores)
        logging.info(f"TRAIN SCORES: {train_scores}")

        scores = evaluate_classification_model(models, test_datasets, test_outputs, feature_names)
        all_test_scores.append(scores)
        logging.info(f"TEST SCORES: {scores}")
        """

    return deltas, all_test_scores, all_train_scores

def run_longer_time_slot(prefix, files):
    deltas = list(range(1, 11))
    all_train_scores, all_test_scores = [], []
    for td in deltas:
        time_delta = td * 1e9
        target_column = 'size'

        # Create paths
        train_paths = []
        test_paths = []
        for f in files:
            train_paths.append(f"{prefix}/{TRAIN}/{f}")
            test_paths.append(f"{prefix}/{TEST}/{f}")
        packets_train = f"{prefix}/{TRAIN}/{PACKETS}"
        packets_test = f"{prefix}/{TEST}/{PACKETS}"

        packets_train_df = get_trace(packets_train, names=['timestamp', 'size', 'src_ip', 'dest_ip', 'src_port', 'dest_port', 'dir'], dtype={'dir': 'int8'})
        packets_train_df = packets_train_df[packets_train_df['src_ip'].isin([MASTER, TARGET_IP]) & packets_train_df['dest_ip'].isin([MASTER, TARGET_IP])]
        logging.info(f"Creating train dataset for {time_delta}")
        train_dataset = create_long_dataset(packets_train_df, train_paths, time_delta)

        packets_test_df = get_trace(packets_test, names=['timestamp', 'size', 'src_ip', 'dest_ip', 'src_port', 'dest_port', 'dir'], dtype={'dir': 'int8'})
        packets_test_df = packets_test_df[packets_test_df['src_ip'].isin([MASTER, TARGET_IP]) & packets_test_df['dest_ip'].isin([MASTER, TARGET_IP])]
        test_dataset = create_long_dataset(packets_test_df, test_paths, time_delta)

        """
        test_outputs = to_em(test_outputs)
        train_outputs = to_em(train_outputs)
        """

        train_dataset.to_csv('now/train/f.csv', index=False)
        test_dataset.to_csv('now/test/f.csv', index=False)
        r, m = xgboost_learn.xgboost_learn(f'now/train/', f'now/test/')
        all_test_scores.append(r['test'])
        all_train_scores.append(r['train'])

        """
        models, train_scores = train_classification_xgb_model(train_datasets, train_outputs, feature_names)
        all_train_scores.append(train_scores)
        logging.info(f"TRAIN SCORES: {train_scores}")

        scores = evaluate_classification_model(models, test_datasets, test_outputs, feature_names)
        all_test_scores.append(scores)
        logging.info(f"TEST SCORES: {scores}")
        """

    return deltas, all_test_scores, all_train_scores


if __name__ == "__main__":
    run_flux_classification(SPARK_PREFIX, files=[digger.ALLOCS, digger.DISK_READS, digger.DISK_WRITES, digger.VIRTUAL_MEMORY, digger.RSS_MEMORY, digger.DATA_MEMORY, digger.S_TIME, digger.U_TIME])
