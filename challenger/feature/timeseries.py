import pandas as pd
import numpy as np
from feature.abstract import Feature
from constants import (
    MAX_LEN
)


class ServerToServerTimeseries(Feature):
    def __init__(
            self,
            packets_df,
            time_delta,
            s_ip,
            d_ips,
            next_start_time=None
    ):
        times, sizes, next_start_time = self.get_flow_idx(
            packets_df, time_delta, s_ip, d_ips, next_start_time
        )
        self.df = self.create(times, sizes)
        self.flow_times = times
        self.next_start_time = next_start_time

    @staticmethod
    def create(times, sizes):
        return pd.DataFrame({
            'start_time': times[:, 0],
            'end_time': times[:, 1],
            'size': sizes,
        }).sort_values(by='start_time')

    def create_feature_df(self):
        return self.df

    def get_flow_idx(
            self, packets_df, time_delta, s_ip, d_ips, next_start_time=None
    ):
        packets_df = packets_df[
            (packets_df['src_ip'] == s_ip) &
            (packets_df['dest_ip'].isin(d_ips))
        ]
        print(len(packets_df))
        if len(packets_df) == 0:
            return (np.empty((0, 2)), np.empty((0,)), next_start_time)
        packets = packets_df[['timestamp', 'size']].values
        if next_start_time is None:
            start = packets_df['timestamp'].min()
        else:
            start = next_start_time
        end = packets_df['timestamp'].max()
        steps_begin = np.arange(start, end, time_delta)
        steps_end = steps_begin + time_delta
        s1 = np.clip(
            np.searchsorted(packets[:, 0], steps_begin), 0, len(packets) - 1
        )
        s2 = np.clip(
            np.searchsorted(packets[:, 0], steps_end), 0, len(packets) - 1
        )
        sizecsum = np.insert(np.cumsum(packets[:, 1]), 0, 0)
        vals = sizecsum[s2] - sizecsum[s1]
        next_start_time = steps_end[-1]
        if len(vals[vals < 0]) > 0:
            import code
            code.interact(local=locals())
        return (
            np.column_stack([steps_begin, steps_end]),
            vals,
            next_start_time
        )
