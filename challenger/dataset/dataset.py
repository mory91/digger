import os
import logging
from utils import force_to_type
import pandas as pd
import numpy as np
from constants import (
    FILES,
    SYSTEM_FEATURES,
    TIME_DELTA,
    NANO_TO_MICRO,
    NROWS,
    B
)
from feature import (
    NetAgg,
    Flow,
    SystemTrace,
    NoFlowException,
    ServerToServerTimeseries,
)
from dataset.abstract import Dataset

netwrok_trace_type_mapping = {
    'timestamp': (1, 'Int64'),
    'size': (2, 'Int32'),
    'src_ip': (3, 'Int16'),
    'dest_ip': (4, 'Int16'),
    'src_port': (5, 'Int32'),
    'dest_port': (6, 'Int32'),
}


def reload_df(f):
    flows_df = pd.read_csv(f, index_col=False)
    cols = flows_df.columns
    flows_df[cols] = flows_df[cols].apply(pd.to_numeric, errors='coerce')
    flows_df = flows_df.dropna()
    return flows_df


class NetworkSeriesDataset(Dataset):
    def __init__(
            self,
            source_ip,
            dest_ips,
            packets_path,
            volumefactor=B,
            chunksize=None,
            time_delta=TIME_DELTA * NANO_TO_MICRO,
            target_file='flows.csv'
    ):
        self.source_ip = source_ip
        self.destination_ips = dest_ips
        self.chunksize = chunksize
        self.target_file = target_file
        self.packets_path = packets_path
        self.time_delta = time_delta
        self.names = sorted(list(netwrok_trace_type_mapping.keys()),
                            key=lambda x: netwrok_trace_type_mapping[x][0])
        self.dtype = {k: v[1] for k, v in netwrok_trace_type_mapping.items()}
        self.volumefactor = volumefactor

    def clean_df(self, df):
        df = force_to_type(df, names=self.names, dtype=self.dtype)
        clean_ips = [self.source_ip] + self.destination_ips
        df = df[
            df['src_ip'].isin(clean_ips) &
            df['dest_ip'].isin(clean_ips)
        ]
        df['size'] = df['size'] / self.volumefactor
        return df

    def get_network_series(
        self,
        packets_df,
        time_delta,
        source,
        destinations,
        start_time
    ):
        pass

    def get_network_aggregated(
        self,
        packets_df,
        times,
        source,
        destinations,
        in_init=0,
        out_init=0,
    ):
        net_agg = NetAgg(
            packets_df, times, source, destinations, in_init, out_init
        )
        ds_net_agg = net_agg.create_feature_df()
        return net_agg, ds_net_agg

    def create_df(self):
        pass

    def packet_to_flow(
        self,
        packets_df,
        time_delta,
        source,
        destinations,
        in_init=0,
        out_init=0,
        start_time=None
    ):
        flow, ds_flow, next_start_time = self.get_network_series(
            packets_df, time_delta, source, destinations, start_time
        )
        net_agg, ds_net_agg = self.get_network_aggregated(
            packets_df, flow.flow_times[:, 0],
            source, destinations, in_init, out_init
        )
        ds = pd.concat((ds_flow, ds_net_agg), axis=1)[1:]
        ds = ds.fillna(0)
        return ds, next_start_time


class FlowDataset(NetworkSeriesDataset):
    def get_network_series(
        self,
        packets_df,
        time_delta,
        source,
        destinations,
        start_time
    ):
        flow = Flow(
            packets_df, time_delta, source, destinations
        )
        ds_flow = flow.create_feature_df()
        return flow, ds_flow, None

    def create_ds(self):
        if os.path.exists(self.target_file):
            return
        remaining_df = pd.DataFrame()
        in_init, out_init = 0, 0
        next_start_time = None
        last_idx = 0
        cycles = 0
        for df in pd.read_csv(
                self.packets_path,
                header=None,
                index_col=False,
                names=self.names,
                sep=',',
                on_bad_lines='skip',
                chunksize=self.chunksize,
        ):
            df = self.clean_df(df)
            df = pd.concat((remaining_df, df), ignore_index=True)
            df = df.sort_values(by='timestamp')

            end_of_flows = np.flatnonzero(
                np.diff(df['timestamp'].values) >= self.time_delta
            )
            if len(end_of_flows) > 0:
                last_idx = end_of_flows[-1]
            else:
                last_idx = len(df)

            current_df = df[:last_idx]
            remaining_df = df[last_idx:]

            try:
                ds, next_start_time = self.packet_to_flow(
                    current_df, self.time_delta, self.source_ip,
                    self.destination_ips, in_init=in_init, out_init=out_init,
                    start_time=next_start_time
                )
            except NoFlowException:
                logging.info("NO FLOWS NOW")
                continue

            if len(ds) == 0:
                print("LEN DS = 0")
                continue

            in_init = ds['networkin'].iloc[-1]
            out_init = ds['networkout'].iloc[-1]

            if cycles == 0:
                ds.to_csv(self.target_file, index=False)
            else:
                ds.to_csv(self.target_file, index=False,
                          mode='a', header=False)

            cycles += 1


class ServerToServerDataset(NetworkSeriesDataset):
    def get_network_series(
        self,
        packets_df,
        time_delta,
        source,
        destinations,
        start_time
    ):
        ts = ServerToServerTimeseries(
            packets_df, time_delta, source, destinations, start_time
        )
        ds_ts = ts.create_feature_df()
        return ts, ds_ts, ts.next_start_time

    def create_ds(self):
        if os.path.exists(self.target_file):
            return
        remaining_df = pd.DataFrame()
        in_init, out_init = 0, 0
        next_start_time = None
        last_idx = 0
        cycles = 0
        for df in pd.read_csv(
                self.packets_path,
                header=None,
                index_col=False,
                names=self.names,
                sep=',',
                on_bad_lines='skip',
                chunksize=self.chunksize,
        ):
            df = self.clean_df(df)
            df = pd.concat((remaining_df, df), ignore_index=True)
            df = df.sort_values(by='timestamp')

            try:
                ds, next_start_time = self.packet_to_flow(
                    df, self.time_delta, self.source_ip, self.destination_ips,
                    in_init=in_init, out_init=out_init,
                    start_time=next_start_time
                )
            except NoFlowException:
                logging.info("NO FLOWS NOW")
                continue

            if len(ds) == 0:
                print("LEN DS = 0")
                continue

            remaining_df = df[df['timestamp'] > next_start_time]

            in_init = ds['networkin'].iloc[-1]
            out_init = ds['networkout'].iloc[-1]

            if cycles == 0:
                ds.to_csv(self.target_file, index=False)
            else:
                ds.to_csv(self.target_file, index=False,
                          mode='a', header=False)

            cycles += 1


class SystemTraceDataset(Dataset):
    def __init__(self, flows_df, trace_files, trace_target_file):
        self.flows_df = flows_df
        self.trace_files = trace_files

    def create_ds(self):
        return SystemTrace.get_system_traces(
            self.flows_df, self.trace_files
        )


class FullDataset(Dataset):
    def __init__(self, time_delta, prefix,
                 source_ip, destination_ip, full_features,
                 network_ds_class=FlowDataset, drop_replace=()):
        self.time_delta = time_delta
        self.prefix = prefix
        self.source_ip = source_ip
        self.destination_ip = destination_ip
        self.full_features = full_features
        self.dataset_dir = f"{FILES}/{int(self.time_delta/1000)}"
        self.dataset_file = f"{self.dataset_dir}/full.csv"
        self.flows_file = f"{self.dataset_dir}/flows.csv"
        self.dataset_df = None
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.drop_replace = drop_replace
        self.network_ds_class = network_ds_class
        if os.path.exists(self.dataset_file):
            self.dataset_df = reload_df(self.dataset_file)

    def create_ds(self):
        if self.dataset_df is not None:
            return self.dataset_df

        packet_file = f"{self.prefix}/packets"
        flow_dataset = self.network_ds_class(
            self.source_ip, self.destination_ip, packet_file,
            chunksize=NROWS, time_delta=self.time_delta,
            target_file=self.flows_file
        )
        flow_dataset.create_ds()
        flows_df = reload_df(self.flows_file)
        system_traces = [trace_name for trace_name in self.full_features
                         if trace_name in SYSTEM_FEATURES]
        trace_files = [
            (trace_name, f"{self.prefix}/{trace_name}")
            for trace_name in system_traces
        ]
        system_trace_dataset = SystemTraceDataset(
            flows_df, trace_files, self.dataset_file
        )
        system_traces = system_trace_dataset.create_ds()
        if self.dataset_df is not None:
            self.dataset_df = pd.concat(
                (self.dataset_df, system_traces),
                axis=1
            )
        else:
            self.dataset_df = pd.concat(
                (flows_df, system_traces),
                axis=1
            )
        self.dataset_df.dropna()
        self.dataset_df.to_csv(self.dataset_file, index=False)
        return self.dataset_df
