import pandas as pd
import numpy as np
from feature.abstract import Feature


class NetAgg(Feature):
    def __init__(
            self, packets_df, flow_times,
            s_ip, d_ips, ingress_init, egress_init
    ):
        ingress_agg, egress_agg = self.get_net_agg(
            packets_df, flow_times, s_ip, d_ips, ingress_init, egress_init
        )
        self.ingress_agg = ingress_agg
        self.egress_agg = egress_agg

    def create_feature_df(self):
        return pd.DataFrame({
            'networkin': self.ingress_agg,
            'networkout': self.egress_agg,
        })

    @staticmethod
    def get_net_agg(
            packets_df, flow_times, s_ip, d_ips, ingress_init, egress_init
    ):
        in_data = packets_df[
            (packets_df['src_ip'] == s_ip) &
            (packets_df['dest_ip'].isin(d_ips))
        ][['timestamp', 'size']].values
        out_data = packets_df[
            (packets_df['src_ip'].isin(d_ips)) &
            (packets_df['dest_ip'] == s_ip)
        ][['timestamp', 'size']].values
        in_data[:, 1] = np.cumsum(in_data[:, 1])
        out_data[:, 1] = np.cumsum(out_data[:, 1])
        vals_in_in = np.searchsorted(in_data[:, 0], flow_times)
        vals_in_in[vals_in_in == len(in_data)] = len(in_data) - 1
        in_agg = in_data[vals_in_in, 1]
        vals_in_out = np.searchsorted(out_data[:, 0], flow_times)
        vals_in_out[vals_in_out == len(out_data)] = len(out_data) - 1
        out_agg = out_data[vals_in_out, 1]
        return in_agg + ingress_init, out_agg + egress_init
