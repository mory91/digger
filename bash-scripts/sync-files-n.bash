#!/bin/bash
rsync -vzSP --files-from=rsync-files morteza@129.128.215.69:ogomon/records ../data/$1/node-1/
rsync -vzSP --files-from=rsync-files morteza@129.128.215.58:ogomon/records ../data/$1/node-2/
rsync -vzSP --files-from=rsync-files morteza@129.128.215.63:ogomon/records ../data/$1/node-3/
