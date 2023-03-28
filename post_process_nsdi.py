import pandas as pd
import numpy as np

from pathlib import Path

NANO_TO_MICRO = 1000
DEFAULT_PREFIX = 'data/24/node-1'

MASTER = 229
TARGET_IP = 237

TRAIN_PATH = 'train'
TEST_PATH = 'test'

DISK_READ = 'disk_read'
DISK_WRITE = 'disk_write'
U_TIME = 'u_time'
S_TIME = 's_time'
CS_TIME = 'cs_time'
CU_TIME = 'cu_time'
PACKETS = 'packets'
SENDS = 'sends'
TARGET_PACKETS = 'target_packets'
TARGET_PACKETS_INGRESS = 'target_packets_ingress'
TARGET_PACKETS_EGRESS = 'target_packets_egress'
MEMORY = 'memory'
ALLOCATIONS = 'allocations'
DATA_MEMORY = 'data_memory'
RSS_MEMORY = 'rss_memory'
FLOW_EXTRACTION_FILE = PACKETS
PACKETS_EGRESS = 'packets_egress'
PACKETS_INGRESS = 'packets_ingress'
FLOW_GAP = 'flow_gap'

EGRESS = 1
INGRESS = 2

def build_ts(file_name):
    sep = ','
    if ALLOCATIONS in file_name or SENDS in file_name:
        sep = '\t'
    df = pd.read_csv(
        file_name, 
        sep=sep, 
        lineterminator='\n', 
        header=None,
        index_col=False,
        names=['timestamp', 'size'], 
        dtype={'timestamp': 'int64', 'size': 'int64'}
    )
    df = df.sort_values(by='timestamp')
    return df.values


def parse_packets_file(file_name):
    df = pd.read_csv(
        file_name, 
        header=None,
        index_col=False,
        names=['timestamp', 'size', 'src', 'dest'], 
        on_bad_lines='warn',
        keep_default_na=False,
    ).dropna()
    src_ip, dest_ip = TARGET_IP, MASTER
    df_egress = df[(df['src'] == src_ip) & (df['dest'] == dest_ip)][["timestamp", "size"]]
    src_ip, dest_ip = MASTER, TARGET_IP
    df_ingress = df[(df['src'] == src_ip) & (df['dest'] == dest_ip)][["timestamp", "size"]]
    df_egress = df_egress.sort_values(by='timestamp')
    df_ingress = df_ingress.sort_values(by='timestamp')
    return df_egress.values, df_ingress.values


def get_flow(trace, time_delta):
    time_stamp = trace[:, 0]
    value_trace = trace[:, 1]
    time_stamp_next = np.roll(time_stamp, -1)
    diffs = (time_stamp_next - time_stamp)
    diffs_high = np.argwhere(diffs > time_delta).squeeze()
    diffs_high_rolled = np.roll(diffs_high, -1).squeeze()
    flows = np.column_stack((diffs_high, diffs_high_rolled))[:-1]
    flows_sizes = np.array([np.sum(value_trace[slice(*f)]) for f in flows])
    flows_times = np.column_stack((time_stamp[diffs_high], time_stamp[diffs_high_rolled]))[:-1]
    return flows_times, flows_sizes, time_stamp[diffs_high][:-1]

def make_tmp_flows(prefix=DEFAULT_PREFIX):
    print("BEGIN")

    trace_names = [DISK_READ, DISK_WRITE, MEMORY, U_TIME, S_TIME, SENDS, ALLOCATIONS]
    cummulative_traces = [PACKETS_EGRESS, PACKETS_INGRESS, TARGET_PACKETS_INGRESS, TARGET_PACKETS_EGRESS]

    path_train = f'{prefix}/{TRAIN_PATH}'
    path_test = f'{prefix}/{TEST_PATH}'
    
    train_packets_egress, train_packets_ingress = parse_packets_file(f"{path_train}/{PACKETS}")
    test_packets_egress, test_packets_ingress = parse_packets_file(f"{path_test}/{PACKETS}")

    train_target_egress, train_target_ingress = parse_packets_file(f"{path_train}/{TARGET_PACKETS}")
    test_target_egress, test_target_ingress = parse_packets_file(f"{path_test}/{TARGET_PACKETS}")

    flow_trace_train = train_packets_egress
    flow_trace_test = test_packets_egress
    train_traces = {}
    test_traces = {}

    print("AFTER PARSING PACKETS FILE")
    print("BEGINING other traces")

    train_traces[TARGET_PACKETS_EGRESS] = train_target_egress
    test_traces[TARGET_PACKETS_EGRESS] = test_target_egress
    train_traces[TARGET_PACKETS_INGRESS] = train_target_ingress
    test_traces[TARGET_PACKETS_INGRESS] = test_target_ingress

    train_traces[PACKETS_EGRESS] = train_packets_egress
    test_traces[PACKETS_EGRESS] = test_packets_egress
    train_traces[PACKETS_INGRESS] = train_packets_ingress
    test_traces[PACKETS_INGRESS] = test_packets_ingress

    for trace_name in trace_names:
        print(trace_name)
        train_traces[trace_name] = build_ts(f"{path_train}/{trace_name}")
        test_traces[trace_name] = build_ts(f"{path_test}/{trace_name}")

    

    print("START BUILDING")

    trace_names = [DISK_READ, DISK_WRITE, MEMORY, PACKETS_EGRESS, PACKETS_INGRESS, U_TIME, S_TIME, TARGET_PACKETS_EGRESS, TARGET_PACKETS_INGRESS, SENDS, ALLOCATIONS]

    for time_delta in range(500, 20000, 250):
        print(time_delta)

        target_prefix = f'nsdi19/data/tmp/{time_delta}'
        target_test = f'{target_prefix}/test'
        target_train = f'{target_prefix}/train'

        for p in [target_test, target_train]:
            Path(p).mkdir(parents=True, exist_ok=True)

        train_flows, train_sizes, diffs_train = get_flow(flow_trace_train, time_delta * NANO_TO_MICRO)
        test_flows, test_sizes, diffs_test = get_flow(flow_trace_test, time_delta * NANO_TO_MICRO)

        train_trace_result_in_flow = {'time': train_flows[:, 0], 'flow_size': train_sizes}
        test_trace_result_in_flow = {'time': test_flows[:, 0], 'flow_size': test_sizes}

        for trace_name in trace_names:
            result = []
            this_trace = train_traces[trace_name][:, 0]
            train_flows_indices = np.searchsorted(this_trace, train_flows)
            if trace_name == ALLOCATIONS:
                print(train_flows_indices[100:150])
            for f in train_flows_indices:
                if trace_name != SENDS:
                    result.append(np.sum(train_traces[trace_name][slice(*f), 1]))
                else:
                    t = train_traces[trace_name][slice(*f), 1]
                    r = np.where(t > 0, t, -1)
                    if len(r) > 0:
                        result.append(r[0])
                    else:
                        result.append(0)
            if trace_name in cummulative_traces:
                result = np.cumsum(result)
            else:
                result = np.array(result)
            train_trace_result_in_flow[trace_name] = result

            result = []
            this_trace = test_traces[trace_name][:, 0]
            test_flows_indices = np.searchsorted(this_trace, test_flows)
            for f in test_flows_indices:
                if trace_name != SENDS:
                    result.append(np.sum(test_traces[trace_name][slice(*f), 1]))
                else:
                    t = test_traces[trace_name][slice(*f), 1]
                    r = np.where(t > 0, t, -1)
                    if len(r) > 0:
                        result.append(r[0])
                    else:
                        result.append(0)
            if trace_name in cummulative_traces:
                result = np.cumsum(result)
            else:
                result = np.array(result)
            test_trace_result_in_flow[trace_name] = result

        test_trace_result_in_flow[FLOW_GAP] = diffs_test
        train_trace_result_in_flow[FLOW_GAP] = diffs_train
        pd.DataFrame(train_trace_result_in_flow).to_csv(f"{target_train}/flows.csv", index=False)
        pd.DataFrame(test_trace_result_in_flow).to_csv(f"{target_test}/flows.csv", index=False)


if __name__ == "__main__":
    make_tmp_flows()
