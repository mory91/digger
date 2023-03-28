#import cudf
#import cupy
import numpy
import pandas
import config

config = config.get_config()
cache = {}


#def get_df_lib():
#    if cache.get('pd') is not None:
#        return cache.get('pd')
#    else:
#        cache['pd'] = cudf if config.glob.GPU else pandas
#        return cache['pd']
#
#
#def get_np_lib():
#    if cache.get('np') is not None:
#        return cache.get('np')
#    else:
#        cache['np'] = cupy if config.glob.GPU else numpy
#        return cache['np']


def force_to_type(df, names, dtype):
    df[names] = df[names].apply(pandas.to_numeric, errors='coerce')
    df = df.astype(dtype)
    df = df.dropna()
    return df


#pd = get_df_lib()
#np = get_np_lib()
