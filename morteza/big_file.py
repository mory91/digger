import logging
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from trace_process import (
    extract_flow_by_ip,
    get_flow,
    get_flow_trace,
    MAX_LEN
)
import xgboost_learn

logging.basicConfig(filename='out.log', encoding='utf-8', level=logging.INFO)

IPS = [237, 212, 144]
MASTER = 237
NODE_2 = 229
TARGET_IP = NODE_2
NODE_3 = 212
NODE_4 = 144

ip_pairs = [
    (NODE_2, MASTER), (NODE_3, MASTER), (NODE_2, MASTER), (NODE_3, MASTER),
    (NODE_2, MASTER), (NODE_3, MASTER), (NODE_2, MASTER), (NODE_3, MASTER),
    (NODE_2, MASTER), (NODE_3, MASTER), (NODE_2, MASTER), (NODE_4, NODE_2),
    (NODE_3, MASTER)
]
tensorflow_ip = (NODE_2, MASTER)

NUMBER_OF_TREES = 50
SHIFT_WINDOW = 5

REGRESSION = 0
CLASSIFICATION = 1

B = 1
MB = 1024 * 1024
KB = 1024
GB = 1024 * 1024 * 1024
NROWS = 30e6
TIME_DELTA = 5000
NANO_TO_MICRO = 1000
NANO_TO_SECONDS = 1e9
NANO_SECONDS = 1
SPARK_PREFIX9 = "../data/29/node-2"
SPARK_PREFIX10 = "../data/29/node-3"

TENSORFLOW_PREFIX1 = "../data/24/node-1"
TENSORFLOW_PREFIX2 = "../data/30/node-1"

MEMCACHED_PREFIX1 = "../data/31/node-1"

REDIS_PREFIX1 = "../data/32/node-1"

SGD_PREFIX1 = "../data/33/node-2"
SGD_PREFIX2 = "../data/34/node-3"
SGD_PREFIX3 = "../data/35/node-3"

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
nsdi_features = [DISK_READS, DISK_WRITES, VIRTUAL_MEMORY, S_TIME]
all_features = [S_TIME, ALLOCS, DISK_READS, DISK_WRITES,
                VIRTUAL_MEMORY, RSS_MEMORY, DATA_MEMORY, U_TIME]
diff_features = list(set(all_features) - set(nsdi_features))
cum_features = [ALLOCS]


def get_net_in_out(
        packets_df,
        flow_times,
        src_ip,
        dest_ip,
        in_init_value=0,
        out_init_value=0
):
    in_data = packets_df[(packets_df['src_ip'] == src_ip) & (
        packets_df['dest_ip'] == dest_ip)][['timestamp', 'size']].values
    out_data = packets_df[(packets_df['src_ip'] == dest_ip) & (
        packets_df['dest_ip'] == src_ip)][['timestamp', 'size']].values
    in_data[:, 1] = np.cumsum(in_data[:, 1])
    out_data[:, 1] = np.cumsum(out_data[:, 1])
    vals_in_in = np.searchsorted(in_data[:, 0], flow_times)
    vals_in_in[vals_in_in == len(in_data)] = len(in_data) - 1
    in_agg = in_data[vals_in_in, 1]
    vals_in_out = np.searchsorted(out_data[:, 0], flow_times)
    vals_in_out[vals_in_out == len(out_data)] = len(out_data) - 1
    out_agg = out_data[vals_in_out, 1]
    return in_agg + in_init_value, out_agg + out_init_value


def get_last_threshed(np_ar, delta):
    time_diffs = np.diff(np_ar)
    return np.flatnonzero(time_diffs >= delta)[-1]


def create_flow_centric_dataset(
        packets_df,
        time_delta,
        source,
        destination,
        in_init=0,
        out_init=0
):
    flows_packets_df = extract_flow_by_ip(packets_df, source, destination)[0]
    flow_time, flow_size = get_flow(flows_packets_df.values, time_delta)
    time_trace, size_trace = get_flow_trace(
        flows_packets_df.values, time_delta, max_len=3)

    flow_gap = np.roll(np.roll(flow_time[:, 0], -1) - flow_time[:, 1], 1)

    fts = flow_time[:, 0]
    in_agg, out_agg = get_net_in_out(
        packets_df, fts, source, destination,
        in_init_value=in_init, out_init_value=out_init
    )

    in_agg += in_init
    out_agg += out_init

    ds = pd.DataFrame({
        'start_time': flow_time[:, 0],
        'end_time': flow_time[:, 1],
        'size': flow_size,
        'gap': flow_gap,
        'networkin': in_agg,
        'networkout': out_agg,
        'packets_1': size_trace[:, 0],
        'packets_2': size_trace[:, 1],
        'packets_3': size_trace[:, 2]
    })[1:]
    ds = ds.fillna(0)

    return ds


def get_trace_error(df, names, dtype, timefactor=NANO_SECONDS, volumefactor=B):
    df[names] = df[names].apply(pd.to_numeric, errors='coerce')
    df = df.astype(dtype)
    df['size'] = df['size'] / volumefactor
    df = df.sort_values(by='timestamp')
    df = df.dropna()
    return df


def create_flows_ds(prefix, ip_pair, time_delta=TIME_DELTA * NANO_TO_MICRO):
    flows_file_name = "flows.csv"
    source, destination = ip_pair
    remaining_df = pd.DataFrame()
    in_init, out_init = 0, 0
    last_idx = 0
    packets_path = f"{prefix}/{TRAIN}/{PACKETS}"
    cycles = 0
    names = ['timestamp', 'size', 'src_ip',
             'dest_ip', 'src_port', 'dest_port', 'dir']
    dtype = {
        'dir': 'Int8', 'timestamp': 'Int64',
        'size': 'Int32', 'src_ip': 'Int16',
        'dest_ip': 'Int16', 'src_port': 'Int32', 'dest_port': 'Int32'
    }
    for df in pd.read_csv(
            packets_path,
            header=None,
            index_col=False,
            names=names,
            sep=',',
            on_bad_lines='skip',
            chunksize=NROWS
    ):
        logging.info("CYCLE")
        packets_df = get_trace_error(df, names=names, dtype=dtype)
        packets_df = pd.concat((remaining_df, packets_df))

        times = packets_df['timestamp'].values
        last_idx = get_last_threshed(times, time_delta)
        packets_df = packets_df[:last_idx]

        packets_df = packets_df[
            packets_df['src_ip'].isin([source, destination]) &
            packets_df['dest_ip'].isin([source, destination])
        ]

        ds = create_flow_centric_dataset(
            packets_df, time_delta, source, destination,
            in_init=in_init, out_init=out_init
        )

        in_init = ds['networkin'].iloc[-1]
        out_init = ds['networkout'].iloc[-1]

        remaining_df = packets_df[last_idx:]

        if cycles == 0:
            ds.to_csv(flows_file_name, index=False)
        else:
            ds.to_csv(flows_file_name, index=False, mode='a', header=False)

        cycles += 1

    return flows_file_name


def get_df_from_file(f):
    flows_df = pd.read_csv(f, index_col=False)
    cols = flows_df.columns
    flows_df[cols] = flows_df[cols].apply(pd.to_numeric, errors='coerce')
    flows_df = flows_df.dropna()
    return flows_df


def run_flux(
        flows_df,
        split=0.3,
        problem=CLASSIFICATION,
        emthresh=3500,
        target_column='size'
):

    flows_df_tmp = flows_df.copy()

    if problem == CLASSIFICATION:
        o = flows_df_tmp[target_column]
        o = np.array(o > emthresh) * 1
        flows_df_tmp[target_column] = o

    flows_train, flows_test = train_test_split(
        flows_df_tmp, shuffle=False, test_size=split
    )

    flows_train.to_csv('now/train/flows.csv', index=False)
    flows_test.to_csv('now/test/flows.csv', index=False)

    r, m = xgboost_learn.xgboost_learn(
        'now/train/', 'now/test/', problem_type=problem, emthresh=emthresh
    )
    return r


def get_system_traces(flows_df, trace_files, cummulative_files, prefix):
    fts = flows_df['start_time'].values
    result = {}
    for trace_name in trace_files:
        trace_path = f"{prefix}/{TRAIN}/{trace_name}"

        sep = ','
        if ALLOCS in trace_name:
            sep = '\t'

        names = ['timestamp', 'size']
        dtype = {'timestamp': 'Int64', 'size': 'Int64'}
        df = pd.read_csv(
            trace_path, names=names, header=None, quoting=3,
            index_col=False, sep=sep, on_bad_lines='skip'
        )
        df = get_trace_error(df, names, dtype)

        values_in_df = np.searchsorted(df['timestamp'].values, fts)
        values_in_df[values_in_df == len(df)] = len(df) - 1

        if trace_name in cummulative_files:
            size_csum = np.cumsum(df['size'])
            trace_values = size_csum[values_in_df].values
        else:
            trace_values = df['size'][values_in_df].values

        result[trace_name] = trace_values

    traces = pd.DataFrame(result)
    full_df = pd.concat((flows_df, traces), axis=1)

    full_df = full_df.dropna()

    full_df.to_csv('full_flows.csv', index=False)
    return full_df


def create_flow_trace_dataset(
        packets_df,
        time_delta,
        source,
        destination,
        max_len=MAX_LEN
):
    flows_packets_df = extract_flow_by_ip(packets_df, source, destination)[0]
    flow_time_trace, flow_size_trace, max_len = get_flow_trace(
        flows_packets_df.values, time_delta, max_len
    )
    time_df_dict = {}
    size_df_dict = {}
    for i in range(flow_time_trace.shape[1]):
        time_df_dict['p' + str(i)] = flow_time_trace[:, i]
    for i in range(flow_size_trace.shape[1]):
        size_df_dict['p' + str(i)] = flow_size_trace[:, i]
    time_df = pd.DataFrame(time_df_dict, index=None)
    size_df = pd.DataFrame(size_df_dict, index=None)
    return time_df, size_df, max_len


def relax_with_maxlen(filename, max_len):
    try:
        df = pd.read_csv(filename, index_col=False)
    except (FileNotFoundError):
        return
    last_col = list(df.keys())[-1]
    last_col = int(last_col[1:])
    last_col += 1
    while last_col < max_len:
        new_df = pd.DataFrame(
            {('p' + str(last_col)): np.zeros(len(df), dtype=np.int32)},
            index=None
        )
        df = pd.concat((df, new_df), axis=1)
        last_col += 1
    df.to_csv(filename, index=False, index_label=False)


def create_flows_trace(prefix, ip_pair, time_delta=TIME_DELTA * NANO_TO_MICRO):
    flows_trace_size_filename = "flows_size_trace.csv"
    flows_trace_time_filename = "flows_time_trace.csv"
    source, destination = ip_pair
    remaining_df = pd.DataFrame(index=None)
    last_idx = 0
    packets_path = f"{prefix}/{TRAIN}/{PACKETS}"
    cycles = 0
    min_time = 0
    names = ['timestamp', 'size', 'src_ip',
             'dest_ip', 'src_port', 'dest_port', 'dir']
    dtype = {
        'dir': 'Int8', 'timestamp': 'Int64',
        'size': 'Int32', 'src_ip': 'Int16',
        'dest_ip': 'Int16', 'src_port': 'Int32', 'dest_port': 'Int32'
    }
    max_len = MAX_LEN
    min_time = None
    for df in pd.read_csv(
            packets_path,
            header=None,
            index_col=False,
            names=names,
            sep=',',
            on_bad_lines='skip',
            chunksize=NROWS
    ):
        packets_df = get_trace_error(df, names=names, dtype=dtype)
        packets_df = pd.concat((remaining_df, packets_df))

        times = packets_df['timestamp'].values

        if min_time is None:
            min_time = np.min(times)

        times = times - min_time
        packets_df['timestamp'] = times

        last_idx = get_last_threshed(times, time_delta)
        packets_df = packets_df[:last_idx]

        packets_df = packets_df[
            packets_df['src_ip'].isin([source, destination]) &
            packets_df['dest_ip'].isin([source, destination])
        ]

        time_df, size_df, new_max_len = create_flow_trace_dataset(
            packets_df, time_delta, source, destination, max_len)
        if new_max_len > max_len and cycles != 0:
            max_len = new_max_len
            relax_with_maxlen(flows_trace_size_filename, max_len)
            relax_with_maxlen(flows_trace_time_filename, max_len)

        remaining_df = packets_df[last_idx:]

        if cycles == 0:
            time_df.to_csv(flows_trace_time_filename,
                           index=False, index_label=False)
            size_df.to_csv(flows_trace_size_filename,
                           index=False, index_label=False)
        else:
            time_df.to_csv(flows_trace_time_filename, index=False,
                           mode='a', index_label=False, header=False)
            size_df.to_csv(flows_trace_size_filename, index=False,
                           mode='a', index_label=False, header=False)

        cycles += 1

    return flows_trace_size_filename, flows_trace_time_filename


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "full":
        full_c_results = []
        nsdi_c_results = []
        full_r_results = []
        nsdi_r_results = []
        times = range(500, 10000, 500)
        for td in times:
            time_delta = td * NANO_TO_MICRO
            logging.info(td)
            flows_file_name = create_flows_ds(
                SGD_PREFIX3, ip_pairs[-1], time_delta
            )
            flows_df = get_df_from_file(flows_file_name)
            featured_flows_df = get_system_traces(
                flows_df, all_features, cum_features, SGD_PREFIX3
            )
            r_c_full = run_flux(featured_flows_df)
            r_r_full = run_flux(featured_flows_df, problem=REGRESSION)
            featured_flows_df = featured_flows_df.drop(columns=diff_features)
            r_c_nsdi = run_flux(featured_flows_df)
            r_r_nsdi = run_flux(featured_flows_df, problem=REGRESSION)
            full_c_results.append(r_c_full)
            nsdi_c_results.append(r_c_nsdi)
            full_r_results.append(r_r_full)
            nsdi_r_results.append(r_r_nsdi)
        logging.info(f"TIMES {times}")
        logging.info(f"FULL_C_RESULTS {full_c_results}")
        logging.info(f"NSDI_C_RESULTS {nsdi_c_results}")
        logging.info(f"FULL_R_RESULTS {full_r_results}")
        logging.info(f"NSDI_R_RESULTS {nsdi_r_results}")
    if cmd == "flow":
        td = int(sys.argv[2])
        flows_file_name = create_flows_ds(
            SGD_PREFIX3, ip_pairs[-1], td * NANO_TO_MICRO
        )
    if cmd == "trace":
        td = int(sys.argv[2])
        time_delta = td * NANO_TO_MICRO
        flows_file_name = create_flows_trace(
            SGD_PREFIX3, ip_pairs[-1], time_delta
        )
