#!/bin/bash
rsync -vzSP --files-from=rsync-files morteza@node-1:ogomon/records ../data/$1/node-1/
