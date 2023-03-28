#!/bin/bash
scp -T morteza@node-1:'~/ogomon/disk_write ~/ogomon/disk_read ~/ogomon/memory ~/ogomon/packets ~/ogomon/cpu_allocations ~/ogomon/data_memory ~/ogomon/rss_memory ~/ogomon/s_time ~/ogomon/u_time ~/ogomon/sends ~/ogomon/kcache' ~/code/digger/data/4/node-1/train/
