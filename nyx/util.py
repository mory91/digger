import pandas as pd
from constants import (
    SECONDS,
    KB,
)


def read_trace_file(file_path, names, dtype, sep=',', sort=True, header=None):
    df = pd.read_csv(
        file_path,
        header=header,
        index_col=False,
        names=names,
        dtype=dtype,
        sep=sep
    )
    if sort:
        df = df.sort_values(by='timestamp')
    return df


def get_packets(file_path, timefactor=SECONDS, volumefactor=KB, sep=',', sort=True):
    names = ['timestamp', 'size', 'src_ip',
             'dest_ip', 'src_port', 'dest_port']
    dtype = None
    df = read_trace_file(file_path, names, dtype, sep=sep, sort=sort)
    df['size'] = df['size'] / volumefactor
    df['timestamp'] = df['timestamp'] / timefactor
    return df


def get_trace(file_path, timefactor=SECONDS, volumefactor=KB, sep=','):
    names = ['timestamp', 'size']
    df = read_trace_file(file_path, names, dtype=None, sep=sep)
    df['size'] = df['size'] / volumefactor
    df['timestamp'] = df['timestamp'] / timefactor
    return df
