#!/bin/bash
rsync -vzSP --files-from=rsync-files morteza@lab-bardia:code/ogomon/records ../data/$1/bardia/
