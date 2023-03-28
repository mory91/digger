#!/bin/bash
scp -T morteza@node-3:'~/ogomon/disk_write ~/ogomon/disk_read ~/ogomon/memory ~/ogomon/packets ~/ogomon/allocations ~/ogomon/data_memory ~/ogomon/rss_memory ~/ogomon/s_time ~/ogomon/u_time ~/ogomon/sends' ~/code/digger/data/40/node-3/train/
