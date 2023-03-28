B = 1
MB = 1024 * 1024
KB = 1024
GB = 1024 * 1024 * 1024
TIME_DELTA = 5000
NANO_TO_MICRO = 1000
NANO_TO_SECONDS = 1e9
NANO_SECONDS = 1
NROWS = 30e6
MAX_LEN = 5
PACKETS = "packets"
SENDS = "sends"
ALLOCS = "cpu_allocations"
CU_ALLOCS = "cuda_allocations"
DISK_READS = "disk_read"
DISK_WRITES = "disk_write"
VIRTUAL_MEMORY = "memory"
RSS_MEMORY = "rss_memory"
DATA_MEMORY = "data_memory"
S_TIME = "s_time"
U_TIME = "u_time"
CS_TIME = "cs_time"
CU_TIME = "cu_time"
PACKETS_1 = "packets_1"
PACKETS_2 = "packets_2"
PACKETS_3 = "packets_3"
PACKETS_4 = "packets_4"
PACKETS_5 = "packets_5"
TIMES_1 = "times_1"
TIMES_2 = "times_2"
TIMES_3 = "times_3"
START_TIME = 'start_time'
END_TIME = 'end_time'
SIZE = 'size'
GAP = 'gap'
NETWORK_IN = 'networkin'
NETWORK_OUT = 'networkout'
START_TIME = 'start_time'
SRC_IP = 'src_ip'
DEST_IP = 'dest_ip'
SRC_PORT = 'src_port'
DEST_PORT = 'dest_port'

NSDI_FEATURES = [DISK_READS, DISK_WRITES, VIRTUAL_MEMORY, S_TIME,
                 START_TIME, END_TIME, GAP, NETWORK_IN, SIZE,
                 NETWORK_OUT, SENDS]  # SRC_IP, DEST_IP, SRC_PORT, DEST_PORT]
ALL_FEATURES = NSDI_FEATURES + [ALLOCS, RSS_MEMORY, DATA_MEMORY, U_TIME, SRC_IP, DEST_IP, SRC_PORT, DEST_PORT]
FS_FEATURES = [SIZE, GAP, DISK_WRITES]
# FS_FEATURES = [SIZE, GAP, DISK_WRITES, RSS_MEMORY, DATA_MEMORY]

"""
NSDI_FEATURES = [DISK_READS, DISK_WRITES, VIRTUAL_MEMORY, S_TIME,
                 START_TIME, END_TIME, GAP, NETWORK_IN, SIZE,
                 NETWORK_OUT]
ALL_FEATURES = NSDI_FEATURES + [RSS_MEMORY, DATA_MEMORY, U_TIME]
"""

CUM_FEATURES = [ALLOCS, CU_ALLOCS]
# FIRSTONE = [SENDS]
FIRSTONE = []
SYSTEM_FEATURES = [SENDS, DISK_READS, DISK_WRITES, VIRTUAL_MEMORY, S_TIME,
                   ALLOCS, RSS_MEMORY, DATA_MEMORY, U_TIME]
NETWORK_FEATURES = [START_TIME, END_TIME, GAP, NETWORK_IN, NETWORK_OUT,
                    PACKETS_1, PACKETS_2, PACKETS_3, PACKETS_4, PACKETS_5]

REGRESSION = 0
CLASSIFICATION = 1