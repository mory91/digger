import pandas as pd
import cupy as cp

from pathlib import Path

def build_ts(file_name):
    series = []
    with open(file_name, "r") as f:
        raw_content = f.read()
        for raw_line in raw_content.split("\n")[:-1]:
            t_stamp, val = raw_line.split('\t')
            series.append((int(t_stamp), int(val)))
    series = sorted(series, key=lambda x: x[0])
    return cp.array(series)

def get_flows(trace, time_delta, target_name='flow_size'):
    time_stamp = trace[:, 0]
    value_trace = trace[:, 1]
    time_stamp_next = cp.roll(time_stamp, -1)
    diffs_micro = (time_stamp_next - time_stamp) / 1000
    diffs_high = cp.argwhere(diffs_micro > time_delta).squeeze()
    diffs_high_rolled = cp.roll(diffs_high, -1).squeeze()
    flows = cp.column_stack((diffs_high, diffs_high_rolled))[:-1]
    flows_times = [time_stamp[f[0]] for f in flows]
    flows_sizes = [cp.sum(value_trace[slice(*f)]) for f in flows]
    data = pd.DataFrame({'time': cp.array(flows_times), target_name: cp.array(flows_sizes)})
    return data

def make_tmp_flows(prefix='data/5/node-1'):
    trace_name = 'network_out'
    path_train = f'{prefix}/train'
    path_test = f'{prefix}/test'
    path_validation = f'{prefix}/validation'
    trace_train = build_ts(f"{path_train}/{trace_name}")
    trace_test = build_ts(f"{path_test}/{trace_name}")
    trace_validation = build_ts(f"{path_validation}/{trace_name}")
    for time_delta in range(500, 5500, 250):
        print(time_delta)
        target_prefix = f'nsdi19/data/tmp/{time_delta}'
        target_test = f'{target_prefix}/test'
        target_train = f'{target_prefix}/train'
        target_validation = f'{target_prefix}/validation'
        for p in [target_test, target_train, target_validation]:
            Path(p).mkdir(parents=True, exist_ok=True)
        get_flows(trace_train, time_delta).to_csv(f'{target_train}/flows.csv', index=False)
        get_flows(trace_test, time_delta).to_csv(f'{target_test}/flows.csv', index=False)
        get_flows(trace_validation, time_delta).to_csv(f'{target_validation}/flows.csv', index=False)

if __name__ == "__main__":
    make_tmp_flows()
