#!/bin/bash
scp -T morteza@node-2:'~/ogomon/records/disk_write ~/ogomon/records/disk_read ~/ogomon/records/memory ~/ogomon/records/packets ~/ogomon/records/allocations ~/ogomon/records/data_memory ~/ogomon/records/rss_memory ~/ogomon/records/s_time ~/ogomon/records/u_time ~/ogomon/records/sends' ~/code/digger/data/3/node-2/
