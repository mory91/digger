import pandas as pd
import numpy as np
from feature.abstract import Feature
from constants import (
    MAX_LEN
)


class Flow(Feature):
    def __init__(
            self,
            packets_df,
            time_delta,
            s_ip,
            d_ips
    ):
        flow_times, flow_sizes, gaps, src_ips, dest_ips, src_ports, dest_ports = self.get_flow_idx(
            packets_df,
            time_delta,
            s_ip,
            d_ips
        )
        self.df = Flow.create(
            flow_times, flow_sizes, gaps, src_ips,
            dest_ips, src_ports, dest_ports
        )
        self.flow_times = self.df[['start_time', 'end_time']].values

    @staticmethod
    def create(
            flow_times, flow_sizes, flow_gaps, src_ips, dest_ips,
            src_ports, dest_ports
    ):
        return pd.DataFrame(
            {
                'start_time': flow_times[:, 0],
                'end_time': flow_times[:, 1],
                'size': flow_sizes,
                'gap': flow_gaps,
                'src_ip': src_ips,
                'dest_ip': dest_ips,
                'src_port': src_ports,
                'dest_port': dest_ports,
            }
            # dtype={
            #     'src_port': int,
            #     'dest_port': int,
            #     'src_ip': int,
            #     'dest_ip': int,
            # }
        ).sort_values(by='start_time')

    def create_feature_df(self):
        return self.df

    @staticmethod
    def get_with_maxlen(trace, start_idx, end_idx, max_len):
        pad_count = np.maximum(max_len - (end_idx - start_idx), 0)
        return np.array(
            [np.pad(trace[s_idx:s_idx + max_len - p_count], (0, p_count))
                for s_idx, p_count in zip(start_idx, pad_count)]
        )

    def get_flow_idx(
            self, packets_df, time_delta, s_ip, d_ips, max_len=MAX_LEN
    ):
        flow_times = []
        value_traces = []
        flow_idxs = []
        flow_sizes = []
        gaps = []
        src_ips, dest_ips = [], []
        src_ports, dest_ports = [], []
        for d_ip in d_ips:
            df_ip = packets_df[
                (packets_df['src_ip'] == s_ip) &
                (packets_df['dest_ip'] == d_ip)
            ]
            for _, flows_packets_df in df_ip.groupby(
                    ['src_port', 'dest_port']
            ):
                trace = flows_packets_df.values
                s_port = flows_packets_df.src_port.values[0]
                d_port = flows_packets_df.dest_port.values[0]
                time_stamp = trace[:, 0]
                value_trace = trace[:, 1]
                time_stamp_next = np.roll(time_stamp, -1)
                diffs = (time_stamp_next - time_stamp)
                diffs_high = np.argwhere(diffs > time_delta).squeeze()
                diffs_high_rolled = np.roll(diffs_high, -1).squeeze()
                flow_idx = np.column_stack(
                    (diffs_high, diffs_high_rolled)
                )[:-1]
                flow_time = Flow.get_flow_times(flow_idx, time_stamp)
                flow_sizes.append(Flow.get_flow_sizes(flow_idx, value_trace))
                gaps.append(Flow.get_flow_gaps(flow_time))
                s = len(flow_time)
                src_ips.append(np.full(s, s_ip, dtype=int))
                dest_ips.append(np.full(s, d_ip, dtype=int))
                src_ports.append(np.full(s, s_port, dtype=int))
                dest_ports.append(np.full(s, d_port, dtype=int))
                flow_times.append(flow_time)
                value_traces.append(value_trace)
                flow_idxs.append(flow_idx)
        return (
            np.concatenate(flow_times),
            np.concatenate(flow_sizes),
            np.concatenate(gaps),
            np.concatenate(src_ips),
            np.concatenate(dest_ips),
            np.concatenate(src_ports),
            np.concatenate(dest_ports),
        )

    @staticmethod
    def get_flow_sizes(flow_idxs, value_traces):
        return np.array(
            [np.sum(value_traces[slice(*f)]) for f in flow_idxs]
        )

    @staticmethod
    def get_flow_times(flow_idxs, time_stamps):
        return np.column_stack(
            (time_stamps[flow_idxs[:, 0]],
             time_stamps[flow_idxs[:, 1] - 1])
        )

    @staticmethod
    def get_flow_gaps(flow_times):
        return np.roll(
            np.roll(flow_times[:, 0], -1) - flow_times[:, 1], 1
        )

    def get_packet_trace(self, max_len=MAX_LEN):
        # BUG: RETURNS NEXT FLOW PACKET SERIES FOR CURRENT FLOW
        start_idx = np.roll(self.flow_idx[:, 0], 1)
        end_idx = np.roll(self.flow_idx[:, 1], 1)
        flow_sizes_packet_trace = Flow.get_with_maxlen(
            self.value_trace, start_idx, end_idx, max_len
        )
        flow_times_packet_trace = Flow.get_with_maxlen(
            self.time_stamp, start_idx, end_idx, max_len
        )
        return flow_sizes_packet_trace, flow_times_packet_trace
