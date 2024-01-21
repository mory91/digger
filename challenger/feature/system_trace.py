import pandas as pd
import numpy as np
from feature.abstract import Feature
from utils import force_to_type

from constants import (
    CUM_FEATURES,
    FIRSTONE,
    EVENT_TRACES
)
trace_dtypes = {'timestamp': 'Int64', 'size': 'Int64'}


class SystemTrace(Feature):
    def __init__(self, flow_times, trace_file_path, trace_name):
        trace_values, delta_till_event = self.create_system_trace_from_flows(
            flow_times, trace_file_path, trace_name
        )
        self.trace_name = trace_name
        self.trace_values = trace_values
        self.delta_till_event = delta_till_event

    def create_feature_df(self):
        trace_df = pd.DataFrame({self.trace_name: self.trace_values})
        time_to_event_df = pd.DataFrame({
            f"tt_{self.trace_name}": self.delta_till_event
        })
        return trace_df, time_to_event_df

    @classmethod
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
            return np.zeros(len(flow_times), dtype=float), np.zeros(len(flow_times), dtype=float)
        names = list(trace_dtypes.keys())
        # CHECK FOR NON LEAKING INFORMATION OF FUTURE TO PAST
        df = force_to_type(df, names, trace_dtypes)

        values_in_df = np.searchsorted(
            df['timestamp'].values, flow_start_times
        )
        values_in_df -= 1

        values_in_df[values_in_df == -1] = 0
        values_in_df[values_in_df == len(df)] = len(df) - 1

        time_from_event = (flow_start_times - df['timestamp'][values_in_df]).values

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
        else:
            trace_values = df['size'][values_in_df].values

        return trace_values, time_from_event

    @staticmethod
    def get_system_traces(flows_df, trace_files):
        ft = flows_df[['start_time', 'end_time']].values
        traces = pd.DataFrame()
        for trace_name, trace_file in trace_files:
            system_trace = SystemTrace(
                ft, trace_file, trace_name
            )
            trace_value, tt_e = system_trace.create_feature_df()
            if trace_name in EVENT_TRACES:
                traces = pd.concat((traces, trace_value, tt_e), axis=1)
            else:
                traces = pd.concat((traces, trace_value), axis=1)
        traces = traces.dropna()
        return traces
