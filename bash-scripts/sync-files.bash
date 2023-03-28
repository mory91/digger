#!/bin/bash
rsync -vzSP --files-from=rsync-files morteza@node-1:ogomon/records ../data/$1/node-1/
rsync -vzSP --files-from=rsync-files morteza@node-3:ogomon/records ../data/$1/node-3/
rsync -vzSP --files-from=rsync-files morteza@node-4:ogomon/records ../data/$1/node-4/
# rsync -vzSP --files-from=rsync-files morteza@node-2:ogomon/records ../data/$1/node-2/
