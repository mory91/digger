from typing import Dict, List
import pandas as pd
import numpy as np

NANO_TO_MICRO = 1000
NANO_TO_SECONDS = 1e9
DEFAULT_PREFIX = 'data/7/node-1'

TRAIN_PATH = 'train'
TEST_PATH = 'test'
VALIDATION_PATH = 'validation'

NETWORK_OUT = 'network_out'
NETWORK_IN = 'network_in'
DISK_READ = 'disk_read'
DISK_WRITE = 'disk_write'
MEMORY = 'memory'
FLOW_EXTRACTION_FILE = NETWORK_OUT

MTU = 1514
B = 1
MB = 1024 * 1024
KB = 1024
GB = 1024 * 1024 * 1024
SECONDS = 1e9
MILLI_SECONDS = 1e6
MICRO_SECONDS = 1e3
NANO_SECONDS = 1
MAX_LEN = 10


def extract_flow_by_ip(packets_df, src_ip, dest_ip):
    flows_packets_df = packets_df[(packets_df['src_ip'] == src_ip) & (
        packets_df['dest_ip'] == dest_ip)]
    return [flows_packets_df]


def get_trace(file_path,  names, dtype, sep=',', timefactor=NANO_SECONDS, volumefactor=B):
    df = pd.read_csv(
        file_path,
        header=None,
        index_col=False,
        names=names,
        dtype=dtype,
        sep=sep
    )
    # df = df[df['dir'] == 1]
    # df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / timefactor
    df['size'] = df['size'] / volumefactor
    df = df.sort_values(by='timestamp')
    return df


def build_array(file_name: str, limit: int = -1) -> np.ndarray:
    series: List[List[int]] = []
    with open(file_name, "r") as f:
        while True:
            raw_content: str = f.readline()
            if not raw_content:
                break
            for raw_line in raw_content.split("\n")[:-1]:
                splitted = raw_line.split('\t')[:limit]
                if splitted[0] != '0':
                    series.append(list(map(int, splitted)))
    series = sorted(series, key=lambda x: x[0])
    np_series = np.array(series, dtype=np.float64)
    return np_series


def save_as_pd(nparr: np.ndarray, columns: List[str], filename: str) -> pd.DataFrame:
    pd_dict: Dict[str, np.ndarray] = {}
    for idx, c in enumerate(columns):
        pd_dict[c] = nparr[:, idx]

    df = pd.DataFrame(pd_dict)
    df.to_csv(filename)

    return df


def extract_same_routes(df: pd.DataFrame) -> List[pd.DataFrame]:
    routes = df[['sport', 'dport']].drop_duplicates().values
    flows_dfs: List[pd.DataFrame] = []
    for r in routes:
        flows_dfs.append(df[(df['sport'] == r[0]) & (df['dport'] == r[1])])
    return flows_dfs


def get_flows_idx(trace, time_delta):
    time_stamp = trace[:, 0]
    time_stamp_next = np.roll(time_stamp, -1)
    diffs = (time_stamp_next - time_stamp)
    diffs_high = np.argwhere(diffs > time_delta).squeeze()
    diffs_high_rolled = np.roll(diffs_high, -1).squeeze()
    flows = np.column_stack((diffs_high, diffs_high_rolled))[:-1]
    return flows


def get_flow(trace, time_delta):
    time_stamp = trace[:, 0]
    value_trace = trace[:, 1]
    flows = get_flows_idx(trace, time_delta)
    flows_sizes = np.array(
        [np.sum(value_trace[slice(*f)]) for f in flows]
    )
    flows_times = np.column_stack(
        (time_stamp[flows[:, 0]], time_stamp[flows[:, 1] - 1])
    )
    return flows_times, flows_sizes


def get_with_maxlen(trace, start_idx, end_idx, max_len):
    pad_count = np.max(max_len - (end_idx - start_idx), 0)
    return np.array(
        [np.pad(np.array(trace[s_idx:s_idx + max_len]), (0, pad_count))
            for s_idx, p_count in zip(start_idx, pad_count)]
    )


def get_flow_trace(trace, time_delta, max_len=MAX_LEN):
    time_stamp = trace[:, 0]
    value_trace = trace[:, 1]
    flows = get_flows_idx(trace, time_delta)
    # flows_len = flows[:, 1] - flows[:, 0]
    # max_len = max(max_len, np.max(flows_len))
    start_idx = flows[:, 0]
    end_idx = flows[:, 1]
    flows_sizes_trace = get_with_maxlen(
        value_trace, start_idx, end_idx, max_len
    )
    flows_times_trace = get_with_maxlen(
        time_stamp, start_idx, end_idx, max_len
    )
    return flows_times_trace, flows_sizes_trace


def get_flows_index(trace, time_delta):
    time_stamp = trace[:, 0]
    value_trace = trace[:, 1]
    time_stamp_next = np.roll(time_stamp, -1)
    diffs = (time_stamp_next - time_stamp)
    diffs_high = np.argwhere(diffs > time_delta).squeeze()
    diffs_high_rolled = np.roll(diffs_high, -1).squeeze()
    flows = np.column_stack((diffs_high, diffs_high_rolled))[:-1]
    flows_sizes = np.array([np.sum(value_trace[slice(*f)]) for f in flows])
    return flows, flows_sizes


def get_flow_trace_time(trace, time_delta):
    time_stamp = trace[:, 0]
    value_trace = trace[:, 1]
    time_stamp_next = np.roll(time_stamp, -1)
    diffs = (time_stamp_next - time_stamp)
    diffs_high = np.argwhere(diffs > time_delta).squeeze()
    diffs_high_rolled = np.roll(diffs_high, -1).squeeze()
    flows = np.column_stack((diffs_high, diffs_high_rolled))[:-1]
    flows_packets = np.array([value_trace[slice(*f)] for f in flows])
    flow_times = np.array([time_stamp[slice(*f)] for f in flows])
    flows_sizes = np.array([np.sum(value_trace[slice(*f)]) for f in flows])
    flows_span = np.array([time_stamp[f] for f in flows])
    return flows_packets, flow_times, flows_sizes, flows_span
