from functools import lru_cache
import pandas as pd
import numpy as np
from feature.abstract import Feature
from utils import force_to_type

from constants import (
    ALLOCS,
    CUM_FEATURES,
    FIRSTONE,
    SENDS
)
trace_dtypes = {'timestamp': 'Int64', 'size': 'Int64'}


class SystemTrace(Feature):
    def __init__(self, flow_times, trace_file_path, trace_name):
        trace_values = self.create_system_trace_from_flows(
            flow_times, trace_file_path, trace_name
        )
        self.trace_name = trace_name
        self.trace_values = trace_values

    def create_feature_df(self):
        return pd.DataFrame({
                self.trace_name: self.trace_values
        })

    @classmethod
    # @lru_cache(maxsize=None)
    def read_trace_file(cls, path, names, sep=','):
        return pd.read_csv(
            path, names=names,
            header=None, quoting=3, index_col=False,
            sep=sep, on_bad_lines='skip'
        )

    def create_system_trace_from_flows(
            self, flow_times, trace_file_path, trace_name
    ):
        flow_start_times = flow_times[:, 0]
        # flow_end_times = flow_times[:, 1]
        names = tuple(trace_dtypes.keys())
        df = SystemTrace.read_trace_file(trace_file_path, names, ',')
        if len(df) == 0:
            return np.zeros(len(flow_times), dtype=float)
        names = list(trace_dtypes.keys())
        df = force_to_type(df, names, trace_dtypes)

        values_in_df = np.searchsorted(
            df['timestamp'].values, flow_start_times
        )
        values_in_df[values_in_df == len(df)] = len(df) - 1

        if trace_name in CUM_FEATURES:
            size_csum = np.cumsum(df['size'])
            trace_values = size_csum[values_in_df].values
        elif trace_name in FIRSTONE:  # TODO: Needs testing
            t_values = []
            remain = df.to_numpy()
            for f in flow_times:
                nparr = remain[:, 0]
                nparr_sizes = remain[:, 1]
                inside = np.argwhere((nparr <= f[0]))
                if len(inside) > 0:
                    t_values.append(nparr_sizes[inside[-1][0]])
                    remain = remain[inside[-1][0]:]  # Is this ok?
                else:
                    t_values.append(0)

            trace_values = np.array(t_values, dtype=float)
            # nparr = df['timestamp'].to_numpy()
            # nparr_sizes = df['size'].to_numpy()
            # a1 = flow_times[:, 0][:, None] < nparr
            # a2 = nparr < flow_times[:, 1][:, None]
            # grid = np.argwhere(a1 & a2)
            # _, first_occ = np.unique(grid[:, 0], return_index=True)
            # result = np.zeros(flow_times.shape[0])
            # result[grid[first_occ][:, 0]] = nparr_sizes[grid[first_occ][:, 1]]
            # trace_values = result
        else:
            trace_values = df['size'][values_in_df].values

        return trace_values

    @staticmethod
    def get_system_traces(flows_df, trace_files):
        ft = flows_df[['start_time', 'end_time']].values
        traces = pd.DataFrame()
        for trace_name, trace_file in trace_files:
            system_trace = SystemTrace(
                ft, trace_file, trace_name
            )
            traces = pd.concat((traces, system_trace.create_feature_df()),
                               axis=1)

        traces = traces.dropna()
        return traces
