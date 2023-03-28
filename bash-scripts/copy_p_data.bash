#!/bin/bash
scp -T ./p_data.txt morteza@node-1:/usr/local/spark/data/mllib/p_data.txt
scp -T ./p_data.txt morteza@node-2:/usr/local/spark/data/mllib/p_data.txt
scp -T ./p_data.txt morteza@node-3:/usr/local/spark/data/mllib/p_data.txt
scp -T ./p_data.txt morteza@node-4:/usr/local/spark/data/mllib/p_data.txt
